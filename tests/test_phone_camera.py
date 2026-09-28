"""Phone camera QR session, input adapter and HTTPS/WebRTC/WebSocket server."""
import asyncio
import fractions
import ssl
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.camera_input import LOCAL, PHONE, PLACEHOLDER, CameraInputRouter, is_phone_source
from assembly.phone_camera import (
    CLOSED, CONNECTED, EXPIRED, LatestFrame, PhoneSession, SessionError, authorize,
)
from assembly.phone_network import LanAddress, normalize_public_url, rank_addresses


def _has(*modules):
    try:
        for module in modules:
            __import__(module)
    except ImportError:
        return False
    return True


HAS_SERVER = _has("aiohttp", "cryptography", "cv2")
HAS_WEBRTC = HAS_SERVER and _has("aiortc", "av")
HAS_QR = _has("segno", "cv2")


class NetworkTests(unittest.TestCase):
    def test_default_route_private_lan_address_is_ranked_first(self):
        ranked = rank_addresses([
            ("vEthernet (WSL)", "172.28.160.1"),
            ("Loopback", "127.0.0.1"),
            ("Ethernet", "169.254.10.2"),
            ("Wi-Fi", "192.168.1.23"),
            ("Tailscale", "100.101.1.2"),
        ], default_ip="192.168.1.23")
        self.assertEqual([item.ip for item in ranked], ["192.168.1.23", "172.28.160.1", "100.101.1.2"])
        self.assertTrue(ranked[0].is_default_route)

    def test_public_url_must_be_absolute(self):
        self.assertEqual(normalize_public_url("https://abc.trycloudflare.com/"), "https://abc.trycloudflare.com")
        with self.assertRaises(ValueError):
            normalize_public_url("abc.example.com")


class SessionTests(unittest.TestCase):
    def test_token_is_bound_to_the_first_device(self):
        session = PhoneSession.create(60, now=100)
        with self.assertRaises(SessionError) as wrong_sid:
            authorize(session, "other", session.token, "a", now=101)
        self.assertEqual(wrong_sid.exception.status, 410)
        with self.assertRaises(SessionError) as wrong_token:
            authorize(session, session.session_id, "bad", "a", now=101)
        self.assertEqual(wrong_token.exception.status, 403)
        authorize(session, session.session_id, session.token, "phone-a", now=101)
        # Reconnect of the same phone is allowed, a second phone is not.
        authorize(session, session.session_id, session.token, "phone-a", now=500)
        with self.assertRaises(SessionError) as taken:
            authorize(session, session.session_id, session.token, "phone-b", now=102)
        self.assertEqual(taken.exception.status, 409)

    def test_unused_qr_expires_and_closed_session_is_rejected(self):
        session = PhoneSession.create(60, now=100)
        with self.assertRaises(SessionError):
            authorize(session, session.session_id, session.token, "a", now=161)
        self.assertEqual(session.state, EXPIRED)
        other = PhoneSession.create(60, now=100)
        other.state = CLOSED
        with self.assertRaises(SessionError) as closed:
            authorize(other, other.session_id, other.token, "a", now=101)
        self.assertEqual(closed.exception.status, 410)

    def test_each_session_has_unique_id_and_token(self):
        a, b = PhoneSession.create(60), PhoneSession.create(60)
        self.assertNotEqual(a.session_id, b.session_id)
        self.assertNotEqual(a.token, b.token)
        self.assertGreaterEqual(len(a.token), 32)


class FakePhone:
    def __init__(self):
        self.frames = LatestFrame()
        self.streaming = False

    def is_streaming(self):
        return self.streaming


class LocalCapture:
    def __init__(self):
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        return True, np.full((20, 30, 3), 7, np.uint8)

    def get(self, prop):
        return 30.0

    def release(self):
        self.released = True


class RouterTests(unittest.TestCase):
    def test_phone_frames_replace_local_source_and_switch_back(self):
        local = LocalCapture()
        router = CameraInputRouter(local, frame_timeout=0.05)
        phone = FakePhone()
        router.attach_phone(phone)
        ok, frame = router.read()
        self.assertTrue(ok)
        self.assertEqual((router.kind, router.generation, frame[0, 0, 0]), (LOCAL, 0, 7))

        phone.streaming = True
        phone.frames.put(np.full((40, 60, 3), 99, np.uint8))
        ok, frame = router.read()
        self.assertEqual((router.kind, router.generation, frame.shape, frame[0, 0, 0]), (PHONE, 1, (40, 60, 3), 99))
        frame[:] = 0  # runtime overlays must not corrupt the buffered frame
        ok, again = router.read()  # no newer frame: last one is repeated
        self.assertEqual(again[0, 0, 0], 99)
        self.assertEqual(local.reads, 1)

        phone.streaming = False
        router.read()
        self.assertEqual((router.kind, router.generation), (LOCAL, 2))
        router.release()
        self.assertTrue(local.released)

    def test_phone_only_source_uses_placeholder_until_connected(self):
        router = CameraInputRouter(None, placeholder_size=(64, 48), placeholder_fps=1000)
        ok, frame = router.read()
        self.assertTrue(ok)
        self.assertEqual((router.kind, frame.shape), (PLACEHOLDER, (48, 64, 3)))
        self.assertTrue(is_phone_source(" Phone "))
        self.assertFalse(is_phone_source("0"))

    def test_latest_frame_waits_for_a_newer_frame(self):
        frames = LatestFrame()
        self.assertIsNone(frames.wait_newer(0, 0.01))
        threading.Timer(0.05, frames.put, args=(np.zeros((2, 2, 3), np.uint8),)).start()
        item = frames.wait_newer(0, 2)
        self.assertIsNotNone(item)
        self.assertEqual(item[0], 1)


@unittest.skipUnless(HAS_QR, "segno/opencv chưa cài")
class QrTests(unittest.TestCase):
    def test_qr_encodes_the_session_url(self):
        import cv2
        from assembly.phone_camera import render_qr

        url = "https://192.168.1.23:8443/phone/abc#token-123"
        image = render_qr(url, 300)
        decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
        self.assertEqual(decoded, url)


@unittest.skipUnless(HAS_SERVER and HAS_QR, "aiohttp/cryptography/segno chưa cài")
class DashboardPhoneTests(unittest.TestCase):
    def test_button_panel_and_status_are_drawn_and_clickable(self):
        from assembly.config import load_config
        from assembly.fsm import ConfigurableAssemblyTracker
        from assembly.hybrid_dashboard import PHONE_BUTTON, PHONE_DISCONNECT, draw_dashboard, phone_hit_test
        from assembly.phone_camera import PhoneCameraStatus, render_qr
        from assembly.phone_camera_ui import DISCONNECT, OPEN, PhoneDashboardView
        from assembly.project_config import build_fusion_engine, load_project_config

        project = load_project_config(ROOT / "configs/projects/earbud_v2.json", ROOT)
        tracker = ConfigurableAssemblyTracker(load_config(project.fsm_config))
        status = PhoneCameraStatus(state="waiting", url="https://192.168.1.2:8443/phone/x#t",
                                   address="192.168.1.2", address_count=2, expires_in=90)
        for panel_open in (False, True):
            view = PhoneDashboardView(panel_open, status, render_qr(status.url), None, PLACEHOLDER)
            screen = draw_dashboard(np.zeros((480, 640, 3), np.uint8), tracker=tracker,
                                    fusion=build_fusion_engine(project.fusion), prediction=None,
                                    prediction_fresh=False, action_threshold=0.5, phone=view)
            self.assertEqual(screen.shape, (900, 1440, 3))
        x = (PHONE_BUTTON[0] + PHONE_BUTTON[2]) // 2
        y = (PHONE_BUTTON[1] + PHONE_BUTTON[3]) // 2
        self.assertEqual(phone_hit_test(x, y, view), OPEN)
        self.assertEqual(phone_hit_test(PHONE_DISCONNECT[0] + 5, PHONE_DISCONNECT[1] + 5, view), DISCONNECT)
        closed = PhoneDashboardView(False, status, None, None, LOCAL)
        self.assertIsNone(phone_hit_test(PHONE_DISCONNECT[0] + 5, PHONE_DISCONNECT[1] + 5, closed))


@unittest.skipUnless(HAS_SERVER, "aiohttp/cryptography chưa cài")
class ServerTests(unittest.TestCase):
    def setUp(self):
        from assembly.phone_camera import PhoneCameraServer

        self.tmp = tempfile.TemporaryDirectory()
        self.server = PhoneCameraServer(
            port=0, transport="auto", tls_dir=self.tmp.name,
            addresses=[LanAddress("127.0.0.1", "test", True)], target_fps=30,
        ).start()
        self.session = self.server.new_session()
        self.base = f"https://127.0.0.1:{self.server.port}"
        self.ssl = ssl.create_default_context()
        self.ssl.check_hostname = False
        self.ssl.verify_mode = ssl.CERT_NONE

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    def run_async(self, coroutine):
        return asyncio.run(asyncio.wait_for(coroutine, 30))

    def wait_until(self, predicate, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def test_qr_url_uses_discovered_address_and_token_fragment(self):
        url = self.server.session_url()
        self.assertEqual(url, f"{self.base}/phone/{self.session.session_id}#{self.session.token}")
        self.assertEqual(self.server.status().state, "waiting")

    def test_page_websocket_fallback_and_session_rules(self):
        import aiohttp
        import cv2

        sid, token = self.session.session_id, self.session.token
        jpeg = cv2.imencode(".jpg", np.full((72, 128, 3), 200, np.uint8))[1].tobytes()

        async def scenario():
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl)) as http:
                async with http.get(f"{self.base}/phone/{sid}") as page:
                    self.assertEqual(page.status, 200)
                    body = await page.text()
                    self.assertIn("getUserMedia", body)
                    self.assertNotIn("__PHONE_CONFIG__", body)
                    self.assertNotIn(token, body)
                self.assertEqual(self.server.status().state, "page_opened")
                async with http.post(f"{self.base}/api/phone/{sid}/offer",
                                     json={"token": "bad", "client_id": "a", "type": "offer", "sdp": "x"}) as bad:
                    self.assertEqual(bad.status, 403)

                ws = await http.ws_connect(f"{self.base}/api/phone/{sid}/ws")
                await ws.send_json({"token": token, "client_id": "phone-a"})
                self.assertEqual(await ws.receive_json(), {"ok": True})
                for _ in range(3):
                    await ws.send_bytes(jpeg)
                    self.assertEqual((await ws.receive()).data, "ok")
                status = self.server.status()
                self.assertTrue(status.connected)
                self.assertEqual((status.state, status.transport, status.frame_size),
                                 (CONNECTED, "WebSocket JPEG", (128, 72)))
                self.assertIsNone(status.url)  # QR hidden once a device owns it

                intruder = await http.ws_connect(f"{self.base}/api/phone/{sid}/ws")
                await intruder.send_json({"token": token, "client_id": "phone-b"})
                self.assertEqual((await intruder.receive_json())["status"], 409)
                await intruder.close()

                await asyncio.get_running_loop().run_in_executor(None, self.server.disconnect)
                closed = await ws.receive()
                self.assertIn(closed.type, {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED})
                self.assertFalse(self.server.is_streaming())
                async with http.post(f"{self.base}/api/phone/{sid}/hangup",
                                     json={"token": token, "client_id": "phone-a"}) as revoked:
                    self.assertEqual(revoked.status, 410)

        self.run_async(scenario())
        fresh = self.server.new_session()
        self.assertNotEqual(fresh.token, token)
        self.assertEqual(self.server.status().state, "waiting")

    @unittest.skipUnless(HAS_WEBRTC, "aiortc chưa cài")
    def test_webrtc_video_reaches_the_frame_buffer(self):
        import aiohttp
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from av import VideoFrame

        class Pattern(VideoStreamTrack):
            async def recv(self):
                pts, time_base = await self.next_timestamp()
                image = np.zeros((240, 320, 3), np.uint8)
                image[:, :160] = (0, 0, 255)
                frame = VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts, frame.time_base = pts, time_base
                return frame

        sid, token = self.session.session_id, self.session.token
        result = {}

        async def scenario():
            peer = RTCPeerConnection()
            peer.addTrack(Pattern())
            await peer.setLocalDescription(await peer.createOffer())
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl)) as http:
                async with http.post(f"{self.base}/api/phone/{sid}/offer", json={
                    "token": token, "client_id": "phone-a",
                    "sdp": peer.localDescription.sdp, "type": "offer"}) as response:
                    self.assertEqual(response.status, 200, await response.text())
                    answer = await response.json()
                await peer.setRemoteDescription(RTCSessionDescription(**answer))
                loop = asyncio.get_running_loop()
                result["streamed"] = await loop.run_in_executor(None, self.wait_until, self.server.is_streaming)
                result["frame"] = self.server.frames.wait_newer(0, 5)
                result["status"] = self.server.status()
                async with http.post(f"{self.base}/api/phone/{sid}/hangup",
                                     json={"token": token, "client_id": "phone-a"}) as response:
                    self.assertEqual(response.status, 200)
            await peer.close()

        self.run_async(scenario())
        self.assertTrue(result["streamed"])
        _, frame = result["frame"]
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertGreater(frame[120, 40, 2], 150)  # red half survived encode/decode
        self.assertLess(frame[120, 280, 2], 100)
        self.assertEqual(result["status"].transport, "WebRTC")
        self.assertEqual(self.server.status().state, "disconnected")

        router = CameraInputRouter(None)
        router.attach_phone(SimpleNamespace(is_streaming=lambda: False, frames=self.server.frames))
        self.assertEqual(router.read()[0], True)


if __name__ == "__main__":
    unittest.main()
