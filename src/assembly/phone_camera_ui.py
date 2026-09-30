"""Glue between the OpenCV dashboard buttons and the phone camera server."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from .camera_input import CameraInputRouter, PHONE
from .phone_camera import (
    PhoneCameraServer, PhoneCameraStatus, PhoneCameraUnavailable, render_qr, terminal_qr,
)

# Dashboard actions (also returned by ``hybrid_dashboard.phone_hit_test``).
OPEN = "phone_open"
NEW_SESSION = "phone_new_session"
DISCONNECT = "phone_disconnect"
CLOSE_PANEL = "phone_close_panel"
NEXT_ADDRESS = "phone_next_address"

KEY_ACTIONS = {ord("p"): OPEN, ord("P"): OPEN, ord("n"): NEW_SESSION, ord("N"): NEW_SESSION,
               ord("d"): DISCONNECT, ord("D"): DISCONNECT}


@dataclass(frozen=True)
class PhoneDashboardView:
    panel_open: bool
    status: PhoneCameraStatus | None
    qr: object | None
    error: str | None
    source_kind: str


class PhoneCameraController:
    def __init__(self, router: CameraInputRouter, server_factory: Callable[[], PhoneCameraServer],
                 log: Callable[[str], None] = print) -> None:
        self.router = router
        self.server_factory = server_factory
        self.server: PhoneCameraServer | None = None
        self.log = log
        self.panel_open = False
        self.error: str | None = None
        self._clicks: deque[tuple[int, int]] = deque(maxlen=8)
        self._qr_cache: tuple[str | None, object] = (None, None)
        self._last_connected = False
        self._last_view: PhoneDashboardView | None = None

    # ------------------------------------------------------------- actions
    def _ensure_server(self) -> PhoneCameraServer | None:
        if self.server is not None:
            return self.server
        try:
            self.server = self.server_factory().start()
        except (PhoneCameraUnavailable, ValueError, OSError) as error:
            self.error = str(error)
            self.log(f"[PHONE] {error}")
            return None
        self.error = None
        self.router.attach_phone(self.server)
        for warning in self.server.status().warnings:
            self.log(f"[PHONE] {warning}")
        return self.server

    def _announce(self) -> None:
        url = self.server.session_url() if self.server else None
        if url:
            self.log(f"[PHONE] Quét QR hoặc mở trên điện thoại: {url}")
            qr = terminal_qr(url)
            if qr:
                self.log(qr)

    def open(self) -> None:
        self.panel_open = True
        server = self._ensure_server()
        if server is None:
            return
        status = server.status()
        if status.state in {"idle", "expired", "closed"}:
            self.new_session()

    def new_session(self) -> None:
        server = self._ensure_server()
        self.panel_open = True
        if server is None:
            return
        server.new_session()
        self._announce()

    def disconnect(self) -> None:
        if self.server is not None:
            self.server.disconnect()
            self.log("[PHONE] Đã ngắt camera điện thoại và thu hồi token phiên.")

    def cycle_address(self) -> None:
        if self.server is not None:
            self.server.cycle_address()
            self._announce()

    def handle_action(self, action: str | None) -> bool:
        handlers = {
            OPEN: self.toggle_panel, NEW_SESSION: self.new_session, DISCONNECT: self.disconnect,
            CLOSE_PANEL: self.close_panel, NEXT_ADDRESS: self.cycle_address,
        }
        handler = handlers.get(action or "")
        if handler is None:
            return False
        handler()
        return True

    def toggle_panel(self) -> None:
        if self.panel_open:
            self.close_panel()
        else:
            self.open()

    def close_panel(self) -> None:
        self.panel_open = False

    def handle_key(self, key: int) -> bool:
        return self.handle_action(KEY_ACTIONS.get(key))

    def on_mouse(self, event, x, y, flags=0, param=None) -> None:
        """``cv2.setMouseCallback`` handler; clicks are processed in ``poll``."""
        import cv2

        if event == cv2.EVENT_LBUTTONUP:
            self._clicks.append((x, y))

    # --------------------------------------------------------------- state
    def poll(self, hit_test: Callable[[int, int, PhoneDashboardView], str | None] | None = None
             ) -> PhoneDashboardView:
        if hit_test is not None and self._last_view is not None:
            while self._clicks:
                self.handle_action(hit_test(*self._clicks.popleft(), self._last_view))
        status = self.server.status() if self.server is not None else None
        connected = bool(status and status.connected)
        if connected != self._last_connected:
            if connected:
                self.log(f"[PHONE] Connected ({status.transport}) – camera điện thoại là nguồn vào pipeline.")
                self.panel_open = False  # give the workflow panel back to the operator
            else:
                self.log(f"[PHONE] Disconnected – {status.message if status else ''}")
            self._last_connected = connected
        url = status.url if status is not None else None
        if url != self._qr_cache[0]:
            self._qr_cache = (url, render_qr(url) if url else None)
        view = PhoneDashboardView(
            panel_open=self.panel_open,
            status=status,
            qr=self._qr_cache[1],
            error=self.error,
            source_kind=self.router.kind,
        )
        self._last_view = view
        return view

    @property
    def using_phone(self) -> bool:
        return self.router.kind == PHONE

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
