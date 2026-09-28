"""Camera input adapter: lets the phone stream act as the pipeline's capture.

``CameraInputRouter`` exposes the small ``cv2.VideoCapture`` surface the
runtimes use (``read``/``get``/``isOpened``/``release``). While a phone is
streaming it returns phone frames; otherwise it returns frames from the
original local camera/video, or a placeholder when the runtime was started
with ``--source phone`` and nothing is connected yet. The detection, action
and fusion code downstream is unchanged.
"""
from __future__ import annotations

import time

PHONE_SOURCE = "phone"

LOCAL = "local"
PHONE = "phone"
PLACEHOLDER = "placeholder"


def is_phone_source(value) -> bool:
    return isinstance(value, str) and value.strip().casefold() == PHONE_SOURCE


class CameraInputRouter:
    def __init__(self, local_capture=None, *, placeholder_size=(1280, 720),
                 placeholder_fps: float = 15.0, frame_timeout: float = 0.25) -> None:
        self.local_capture = local_capture
        self.phone = None  # PhoneCameraServer, attached lazily
        self.placeholder_size = placeholder_size
        self.placeholder_interval = 1.0 / placeholder_fps
        self.frame_timeout = frame_timeout
        self.kind = LOCAL if local_capture is not None else PLACEHOLDER
        # Incremented on every source switch so callers can drop temporal
        # state (embeddings, stable-frame counters) that belongs to the old view.
        self.generation = 0
        self._phone_seq = 0
        self._last_phone_frame = None
        self._last_placeholder = 0.0

    def attach_phone(self, server) -> None:
        self.phone = server

    def _switch(self, kind: str) -> None:
        if kind != self.kind:
            self.kind = kind
            self.generation += 1
            if kind != PHONE:
                self._last_phone_frame = None

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return True

    def read(self):
        phone = self.phone
        if phone is not None and phone.is_streaming():
            item = phone.frames.wait_newer(self._phone_seq, self.frame_timeout)
            if item is not None:
                self._phone_seq, frame = item
                self._last_phone_frame = frame
                self._switch(PHONE)
                # Copy: the runtime draws overlays onto the frame it receives.
                return True, frame.copy()
            if self._last_phone_frame is not None and phone.is_streaming():
                return True, self._last_phone_frame.copy()
        if self.local_capture is not None:
            self._switch(LOCAL)
            return self.local_capture.read()
        self._switch(PLACEHOLDER)
        return True, self._placeholder()

    def _placeholder(self):
        import cv2
        import numpy as np

        wait = self.placeholder_interval - (time.perf_counter() - self._last_placeholder)
        if wait > 0:
            time.sleep(wait)
        self._last_placeholder = time.perf_counter()
        width, height = self.placeholder_size
        frame = np.full((height, width, 3), 32, dtype=np.uint8)
        cv2.putText(frame, "Waiting for phone camera - scan the QR code",
                    (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 210, 230), 2, cv2.LINE_AA)
        return frame

    def get(self, prop):
        import cv2

        if self.kind == LOCAL and self.local_capture is not None:
            return self.local_capture.get(prop)
        size = (self._last_phone_frame.shape[1::-1] if self._last_phone_frame is not None
                else self.placeholder_size)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(size[1])
        if prop == cv2.CAP_PROP_FPS:
            return float(self.phone.frames.fps) if self.phone is not None else 0.0
        return 0.0

    def set(self, prop, value) -> bool:
        return self.local_capture.set(prop, value) if self.local_capture is not None else False

    def release(self) -> None:
        if self.local_capture is not None:
            self.local_capture.release()
