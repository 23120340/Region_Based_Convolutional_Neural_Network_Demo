"""Phone camera server: QR session + WebRTC (WebSocket JPEG fallback).

The PC runs a small HTTPS server in a background thread. A QR code points the
phone at ``https://<lan-ip>:<port>/phone/<session_id>#<token>``; the page asks
for camera permission, shows a preview and streams video back:

* WebRTC is used first because the hybrid runtime needs continuous real-time
  video (YOLO every few frames plus a DINOv2/BiLSTM temporal window).
* If ICE cannot connect (Wi-Fi client isolation, tunnel without TURN, old
  browser), the page falls back to JPEG frames over a WebSocket on the same
  HTTPS origin, with one frame in flight at a time (no polling, no backlog).

Only the latest decoded frame is kept; :mod:`assembly.camera_input` turns it
into a ``cv2.VideoCapture``-like source for the existing pipeline.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import ipaddress
import json
import secrets
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT
from .phone_network import LanAddress, discover_lan_addresses, normalize_public_url


PAGE_TEMPLATE = Path(__file__).with_name("phone_camera_page.html")
DEFAULT_TLS_DIR = PROJECT_ROOT / "artifacts" / "phone_camera" / "tls"
TRANSPORTS = ("auto", "webrtc", "websocket")
# A phone is "Connected" only while frames keep arriving.
FRAME_STALE_SECONDS = 3.0

# Session states shown on the PC dashboard.
IDLE = "idle"
WAITING = "waiting"            # QR shown, nobody opened it yet
PAGE_OPENED = "page_opened"    # phone loaded the page, camera not streaming
CONNECTING = "connecting"
CONNECTED = "connected"
DISCONNECTED = "disconnected"  # device dropped; the same device may reconnect
EXPIRED = "expired"            # QR not used before its deadline
CLOSED = "closed"              # revoked from the PC; a new QR is required
ERROR = "error"


class PhoneCameraUnavailable(RuntimeError):
    """Raised when optional dependencies or the network are missing."""


def _require_dependencies() -> None:
    missing = []
    for module in ("aiohttp", "cryptography", "cv2", "numpy"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise PhoneCameraUnavailable(
            "Thiếu thư viện cho camera điện thoại: " + ", ".join(missing)
            + ". Chạy: python -m pip install -r requirements-camera.txt"
        )


def webrtc_available() -> bool:
    try:
        import aiortc  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- TLS
def ensure_self_signed_cert(directory: Path, hosts: list[str]) -> tuple[Path, Path]:
    """Create (or reuse) a certificate whose SAN covers every advertised host.

    Phones only expose ``getUserMedia`` to secure origins, so the LAN page has
    to be HTTPS. The browser will warn once because the certificate is
    self-signed; ``--phone-cert/--phone-key`` accept a trusted one instead.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / "phone_camera.crt", directory / "phone_camera.key"
    wanted = set(hosts) | {"localhost", "127.0.0.1"}
    if cert_path.is_file() and key_path.is_file():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            covered = {str(v) for v in san.get_values_for_type(x509.IPAddress)}
            covered |= set(san.get_values_for_type(x509.DNSName))
            expires = cert.not_valid_after_utc
            if wanted <= covered and expires - dt.timedelta(days=7) > dt.datetime.now(dt.timezone.utc):
                return cert_path, key_path
        except (ValueError, x509.ExtensionNotFound):
            pass

    key = ec.generate_private_key(ec.SECP256R1())
    names: list[x509.GeneralName] = []
    for host in sorted(wanted):
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Assembly Monitor phone camera")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=397))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


# ----------------------------------------------------------------------- session
@dataclass
class PhoneSession:
    """One QR code. Its token is required by every API call of the phone."""

    session_id: str
    token: str
    created_at: float
    expires_at: float
    state: str = WAITING
    client_id: str | None = None
    device: str = ""
    remote: str = ""
    transport: str = ""
    message: str = "Quét QR bằng điện thoại cùng mạng Wi-Fi/LAN."

    @classmethod
    def create(cls, ttl_seconds: float, now: float | None = None) -> "PhoneSession":
        now = time.time() if now is None else now
        return cls(
            session_id=secrets.token_urlsafe(9),
            token=secrets.token_urlsafe(24),
            created_at=now,
            expires_at=now + ttl_seconds,
        )

    @property
    def is_open(self) -> bool:
        return self.state not in {EXPIRED, CLOSED}


class SessionError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def authorize(session: PhoneSession | None, session_id: str, token: str, client_id: str,
              now: float | None = None) -> PhoneSession:
    """Check a phone request and bind the session to the first device using it."""

    now = time.time() if now is None else now
    if session is None or not hmac.compare_digest(session.session_id, str(session_id)):
        raise SessionError(410, "QR này không còn hiệu lực. Nhấn \"QR mới\" trên PC và quét lại.")
    if session.state == WAITING or session.state == PAGE_OPENED:
        if session.client_id is None and now > session.expires_at:
            session.state = EXPIRED
    if not session.is_open:
        raise SessionError(410, "Phiên kết nối đã kết thúc. Tạo QR mới trên PC.")
    if not token or not hmac.compare_digest(session.token, str(token)):
        raise SessionError(403, "Sai token phiên. Hãy quét lại QR đang hiển thị trên PC.")
    if not client_id or len(client_id) > 128:
        raise SessionError(400, "Thiếu client id.")
    if session.client_id is None:
        session.client_id = client_id
    elif not hmac.compare_digest(session.client_id, client_id):
        raise SessionError(409, "Phiên này đã được một điện thoại khác sử dụng.")
    return session


@dataclass(frozen=True)
class PhoneCameraStatus:
    """Thread-safe snapshot for the dashboard."""

    state: str = IDLE
    url: str | None = None
    message: str = ""
    connected: bool = False
    fps: float = 0.0
    frame_size: tuple[int, int] | None = None
    transport: str = ""
    device: str = ""
    address: str = ""
    address_count: int = 0
    expires_in: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ------------------------------------------------------------------------- frame
class LatestFrame:
    """Single-slot frame buffer: newest frame wins, readers can wait for it."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame = None
        self._seq = 0
        self._timestamp = 0.0
        self._fps = 0.0

    def put(self, frame) -> None:
        now = time.monotonic()
        with self._condition:
            if self._timestamp:
                instant = 1.0 / max(now - self._timestamp, 1e-3)
                self._fps = instant if self._fps == 0 else 0.85 * self._fps + 0.15 * instant
            self._frame, self._timestamp = frame, now
            self._seq += 1
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._frame, self._timestamp, self._fps = None, 0.0, 0.0
            self._seq += 1
            self._condition.notify_all()

    def wait_newer(self, seq: int, timeout: float):
        """Return ``(seq, frame)`` newer than ``seq`` or ``None`` on timeout."""
        with self._condition:
            self._condition.wait_for(lambda: self._seq > seq and self._frame is not None, timeout)
            if self._seq > seq and self._frame is not None:
                return self._seq, self._frame
            return None

    def age(self) -> float:
        with self._condition:
            return time.monotonic() - self._timestamp if self._timestamp else float("inf")

    @property
    def fps(self) -> float:
        return self._fps if self.age() < FRAME_STALE_SECONDS else 0.0

    @property
    def shape(self) -> tuple[int, int] | None:
        with self._condition:
            return None if self._frame is None else tuple(self._frame.shape[1::-1])


# ------------------------------------------------------------------------ server
class PhoneCameraServer:
    """HTTPS + WebRTC signalling server running on a private asyncio loop."""

    def __init__(
        self,
        *,
        port: int = 8443,
        host: str | None = None,
        public_url: str | None = None,
        transport: str = "auto",
        cert_file: str | Path | None = None,
        key_file: str | Path | None = None,
        tls_dir: str | Path = DEFAULT_TLS_DIR,
        ice_servers: tuple[str, ...] = (),
        session_ttl: float = 600.0,
        target_size: tuple[int, int] = (1280, 720),
        target_fps: int = 30,
        jpeg_quality: float = 0.8,
        addresses: list[LanAddress] | None = None,
    ) -> None:
        _require_dependencies()
        if transport not in TRANSPORTS:
            raise ValueError(f"transport phải là một trong {TRANSPORTS}")
        if (cert_file is None) != (key_file is None):
            raise ValueError("--phone-cert và --phone-key phải đi cùng nhau")
        self.requested_port = port
        self.port = port
        self.public_url = normalize_public_url(public_url) if public_url else None
        self.transport = transport if transport != "auto" or webrtc_available() else "websocket"
        self.cert_file, self.key_file = cert_file, key_file
        self.tls_dir = Path(tls_dir)
        self.ice_servers = tuple(ice_servers)
        self.session_ttl = session_ttl
        self.target_size = target_size
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.frames = LatestFrame()

        self._lock = threading.RLock()
        self._session: PhoneSession | None = None
        self._peer = None               # RTCPeerConnection
        self._socket = None             # aiohttp WebSocketResponse
        self._tasks: set[asyncio.Task] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner = None
        self._warnings: list[str] = []
        if transport in {"auto", "webrtc"} and not webrtc_available():
            self._warnings.append("Chưa cài aiortc: dùng WebSocket JPEG thay cho WebRTC.")
            if transport == "webrtc":
                raise PhoneCameraUnavailable("--phone-transport webrtc cần aiortc (requirements-camera.txt).")

        if host:
            self.addresses = [LanAddress(host, "--phone-host", True)]
        elif addresses is not None:
            self.addresses = list(addresses)
        else:
            self.addresses = discover_lan_addresses()
        self._address_index = 0
        if not self.addresses and not self.public_url:
            self._warnings.append(
                "PC không có IP LAN nào điện thoại truy cập được. Kết nối PC và điện thoại "
                "vào cùng Wi-Fi/hotspot, hoặc dùng --phone-public-url với tunnel HTTPS."
            )
        elif self.addresses and not self.addresses[0].is_private_lan and not host:
            self._warnings.append(
                f"IP {self.addresses[0].ip} không phải dải LAN riêng; điện thoại có thể không truy cập được."
            )

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "PhoneCameraServer":
        if self._thread is not None:
            return self
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._start_site())
            except BaseException as error:  # reported to the caller thread
                failure.append(error)
                ready.set()
                loop.close()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.run_until_complete(self._shutdown())
                loop.close()

        self._thread = threading.Thread(target=run, name="phone-camera-server", daemon=True)
        self._thread.start()
        ready.wait(20)
        if failure:
            self._thread = None
            raise PhoneCameraUnavailable(f"Không khởi động được server camera điện thoại: {failure[0]}")
        return self

    def stop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        self._thread = None
        self._loop = None

    def _ssl_context(self) -> ssl.SSLContext:
        if self.cert_file is not None:
            cert, key = Path(self.cert_file), Path(self.key_file)
        else:
            hosts = [item.ip for item in self.addresses] + [socket.gethostname()]
            cert, key = ensure_self_signed_cert(self.tls_dir, hosts)
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(str(cert), str(key))
        return context

    async def _start_site(self) -> None:
        from aiohttp import web

        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/healthz", self._handle_health)
        app.router.add_get("/phone/{sid}", self._handle_page)
        app.router.add_post("/api/phone/{sid}/offer", self._handle_offer)
        app.router.add_post("/api/phone/{sid}/hangup", self._handle_hangup)
        app.router.add_get("/api/phone/{sid}/ws", self._handle_websocket)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        context = self._ssl_context()
        last_error: OSError | None = None
        # A busy port should not block the demo: try the next few ports.
        for port in range(self.requested_port, self.requested_port + 10) if self.requested_port else [0]:
            site = web.TCPSite(self._runner, "0.0.0.0", port, ssl_context=context)
            try:
                await site.start()
            except OSError as error:
                last_error = error
                continue
            server = getattr(site, "_server", None)
            sockets = getattr(server, "sockets", None) or []
            self.port = sockets[0].getsockname()[1] if sockets else port
            return
        raise last_error or OSError("Không mở được cổng")

    async def _shutdown(self) -> None:
        await self._close_transport()
        if self._runner is not None:
            await self._runner.cleanup()

    # --------------------------------------------------------------- control
    @property
    def base_url(self) -> str | None:
        if self.public_url:
            return self.public_url
        if not self.addresses:
            return None
        host = self.addresses[self._address_index % len(self.addresses)].ip
        return f"https://{host}:{self.port}"

    def session_url(self, session: PhoneSession | None = None) -> str | None:
        session = session or self._session
        base = self.base_url
        if session is None or base is None:
            return None
        # The token lives in the fragment: browsers never send it in the page
        # request or Referer, only the page script reads it.
        return f"{base}/phone/{session.session_id}#{session.token}"

    def new_session(self) -> PhoneSession:
        """Revoke the current QR/device and issue a fresh session + token."""
        self.start()
        session = PhoneSession.create(self.session_ttl)
        with self._lock:
            previous = self._session
            if previous is not None:
                previous.state = CLOSED
            self._session = session
        self._call(self._close_transport())
        return session

    def disconnect(self) -> None:
        """Drop the phone and revoke its token; the next connection needs a new QR."""
        with self._lock:
            if self._session is not None and self._session.is_open:
                self._session.state = CLOSED
                self._session.message = "Đã ngắt kết nối từ PC. Tạo QR mới để kết nối lại."
        self._call(self._close_transport())

    def cycle_address(self) -> None:
        with self._lock:
            if len(self.addresses) > 1:
                self._address_index = (self._address_index + 1) % len(self.addresses)

    def _call(self, coroutine, timeout: float = 5.0) -> None:
        if self._loop is None:
            coroutine.close()
            return
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            future.result(timeout)
        except Exception:
            pass

    def is_streaming(self) -> bool:
        with self._lock:
            session = self._session
            live = session is not None and session.state == CONNECTED
        return live and self.frames.age() < FRAME_STALE_SECONDS

    def status(self) -> PhoneCameraStatus:
        with self._lock:
            session = self._session
            if session is not None and session.state in {WAITING, PAGE_OPENED} \
                    and session.client_id is None and time.time() > session.expires_at:
                session.state = EXPIRED
                session.message = "QR đã hết hạn. Nhấn \"QR mới\"."
            address = (self.addresses[self._address_index % len(self.addresses)].ip
                       if self.addresses else "")
            if session is None:
                return PhoneCameraStatus(address=address, address_count=len(self.addresses),
                                         warnings=tuple(self._warnings), transport=self.transport)
            state, message = session.state, session.message
            if state == CONNECTED and self.frames.age() >= FRAME_STALE_SECONDS:
                message = "Đã kết nối nhưng chưa nhận frame (điện thoại khóa màn hình?)."
            return PhoneCameraStatus(
                state=state,
                url=self.session_url(session) if session.is_open and session.client_id is None else None,
                message=message,
                connected=self.is_streaming(),
                fps=self.frames.fps,
                frame_size=self.frames.shape,
                transport=session.transport,
                device=session.device,
                address=self.public_url or address,
                address_count=len(self.addresses),
                expires_in=(max(0.0, session.expires_at - time.time())
                            if session.client_id is None and session.is_open else None),
                warnings=tuple(self._warnings),
            )

    # ------------------------------------------------------------- internals
    def _set_state(self, session: PhoneSession, state: str, message: str) -> None:
        with self._lock:
            if session is self._session and session.is_open:
                session.state, session.message = state, message

    async def _close_transport(self) -> None:
        peer, socket_ = self._peer, self._socket
        self._peer = self._socket = None
        for task in list(self._tasks):
            task.cancel()
        if peer is not None:
            await peer.close()
        if socket_ is not None and not socket_.closed:
            await socket_.close(code=4000, message=b"session closed")
        self.frames.clear()

    def _spawn(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _authorize(self, sid: str, token: str, client_id: str) -> PhoneSession:
        with self._lock:
            return authorize(self._session, sid, token, client_id)

    @staticmethod
    def _error(error: SessionError):
        from aiohttp import web

        return web.json_response({"error": error.message}, status=error.status)

    def _page_config(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "iceServers": [{"urls": url} for url in self.ice_servers],
            "width": self.target_size[0],
            "height": self.target_size[1],
            "fps": self.target_fps,
            "jpegQuality": self.jpeg_quality,
        }

    async def _handle_index(self, request):
        from aiohttp import web

        return web.Response(text="Assembly Monitor: quét QR trên màn hình PC để kết nối camera.",
                            content_type="text/plain", charset="utf-8")

    async def _handle_health(self, request):
        from aiohttp import web

        return web.json_response({"ok": True})

    async def _handle_page(self, request):
        from aiohttp import web

        sid = request.match_info["sid"]
        with self._lock:
            session = self._session
            valid = session is not None and session.session_id == sid and session.is_open
            if valid and session.state == WAITING:
                session.state = PAGE_OPENED
                session.message = "Điện thoại đã mở trang, đang chờ cấp quyền camera."
        html = PAGE_TEMPLATE.read_text(encoding="utf-8").replace(
            "__PHONE_CONFIG__", json.dumps(self._page_config()))
        return web.Response(
            text=html, content_type="text/html", charset="utf-8", status=200 if valid else 410,
            headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        )

    async def _read_json(self, request) -> dict:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise SessionError(400, "JSON không hợp lệ.")
        if not isinstance(payload, dict):
            raise SessionError(400, "JSON không hợp lệ.")
        return payload

    async def _handle_offer(self, request):
        from aiohttp import web

        if self.transport == "websocket":
            return web.json_response({"error": "WebRTC đang tắt trên PC.", "fallback": "websocket"}, status=501)
        try:
            payload = await self._read_json(request)
            session = self._authorize(request.match_info["sid"], payload.get("token", ""),
                                      payload.get("client_id", ""))
            if payload.get("type") != "offer" or not isinstance(payload.get("sdp"), str):
                raise SessionError(400, "SDP offer không hợp lệ.")
        except SessionError as error:
            return self._error(error)

        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        from aiortc.mediastreams import MediaStreamError

        await self._close_transport()
        config = RTCConfiguration([RTCIceServer(urls=url) for url in self.ice_servers])
        peer = RTCPeerConnection(configuration=config)
        self._peer = peer
        with self._lock:
            session.device = request.headers.get("User-Agent", "")[:160]
            session.remote = request.remote or ""
            session.transport = "WebRTC"
        self._set_state(session, CONNECTING, "Đang thiết lập WebRTC...")

        async def consume(track) -> None:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    break
                if self._peer is not peer:
                    break
                image = frame.to_ndarray(format="bgr24")
                self.frames.put(image)
                if session.state != CONNECTED:
                    self._set_state(session, CONNECTED, "Đang nhận video realtime qua WebRTC.")

        @peer.on("track")
        def on_track(track) -> None:
            if track.kind == "video":
                self._spawn(consume(track))

        @peer.on("connectionstatechange")
        async def on_state() -> None:
            state = peer.connectionState
            if state in {"failed", "closed"} and self._peer is peer:
                self._set_state(session, DISCONNECTED,
                                "Mất kết nối WebRTC. Điện thoại sẽ thử kết nối lại.")
                self._peer = None
                self.frames.clear()
                await peer.close()

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type="offer"))
            await peer.setLocalDescription(await peer.createAnswer())
        except Exception as error:  # malformed SDP or unsupported codec
            await peer.close()
            self._peer = None
            self._set_state(session, DISCONNECTED, f"WebRTC lỗi: {error}")
            return web.json_response({"error": f"WebRTC lỗi: {error}", "fallback": "websocket"}, status=400)
        return web.json_response({"sdp": peer.localDescription.sdp, "type": peer.localDescription.type})

    async def _handle_hangup(self, request):
        from aiohttp import web

        try:
            payload = await self._read_json(request)
            session = self._authorize(request.match_info["sid"], payload.get("token", ""),
                                      payload.get("client_id", ""))
        except SessionError as error:
            return self._error(error)
        await self._close_transport()
        self._set_state(session, DISCONNECTED, "Điện thoại đã dừng gửi camera.")
        return web.json_response({"ok": True})

    async def _handle_websocket(self, request):
        from aiohttp import WSMsgType, web
        import cv2
        import numpy as np

        socket_ = web.WebSocketResponse(heartbeat=10, max_msg_size=6 * 1024 * 1024)
        await socket_.prepare(request)
        try:
            hello = await socket_.receive_json(timeout=10)
            session = self._authorize(request.match_info["sid"], hello.get("token", ""),
                                      hello.get("client_id", ""))
        except SessionError as error:
            await socket_.send_json({"error": error.message, "status": error.status})
            await socket_.close(code=4000 + error.status)
            return socket_
        except (asyncio.TimeoutError, TypeError, ValueError, AttributeError):
            await socket_.close(code=4400)
            return socket_

        await self._close_transport()
        self._socket = socket_
        with self._lock:
            session.device = request.headers.get("User-Agent", "")[:160]
            session.remote = request.remote or ""
            session.transport = "WebSocket JPEG"
        self._set_state(session, CONNECTING, "Đang chờ frame qua WebSocket...")
        await socket_.send_json({"ok": True})
        async for message in socket_:
            if message.type != WSMsgType.BINARY:
                continue
            if self._socket is not socket_:
                break
            image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                self.frames.put(image)
                if session.state != CONNECTED:
                    self._set_state(session, CONNECTED, "Đang nhận frame qua WebSocket (fallback).")
            # One frame in flight: the phone waits for this ack before sending.
            await socket_.send_str("ok")
        if self._socket is socket_:
            self._socket = None
            self.frames.clear()
            self._set_state(session, DISCONNECTED, "Điện thoại đã ngắt WebSocket.")
        return socket_


def render_qr(data: str, size: int = 280):
    """Return a BGR ndarray QR code, or ``None`` if ``segno`` is missing."""
    try:
        import segno
    except ImportError:
        return None
    import cv2
    import numpy as np

    matrix = np.array(segno.make(data, error="m").matrix, dtype=np.uint8)
    matrix = np.pad(matrix, 4)  # quiet zone
    image = np.where(matrix[..., None] > 0, 0, 255).astype(np.uint8).repeat(3, axis=2)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)


def terminal_qr(data: str) -> str:
    try:
        import segno
    except ImportError:
        return ""
    import io

    buffer = io.StringIO()
    segno.make(data, error="m").terminal(out=buffer, compact=True)
    return buffer.getvalue()


__all__ = [
    "CLOSED", "CONNECTED", "CONNECTING", "DISCONNECTED", "ERROR", "EXPIRED", "IDLE",
    "PAGE_OPENED", "WAITING", "LatestFrame", "PhoneCameraServer", "PhoneCameraStatus",
    "PhoneCameraUnavailable", "PhoneSession", "SessionError", "authorize",
    "ensure_self_signed_cert", "render_qr", "terminal_qr", "webrtc_available",
]
