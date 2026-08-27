from __future__ import annotations

from pathlib import Path
from typing import Any

from .camera_config import CameraConfig
from .vision import Detection


class YoloWorldDetector:
    """Small adapter around Ultralytics YOLO-World with custom text prompts."""

    def __init__(
        self,
        config: CameraConfig,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO, YOLOWorld
        except ImportError as error:
            raise RuntimeError(
                "Thiếu ultralytics. Chạy: python -m pip install -r requirements-camera.txt"
            ) from error

        self.config = config
        self.device = device
        selected_model = model_path or config.model
        self.open_vocabulary = "world" in Path(selected_model).name.lower()
        if self.open_vocabulary:
            self.model = YOLOWorld(selected_model)
            self.model.set_classes(config.prompts)
        else:
            self.model = YOLO(selected_model)

    def predict(self, frame: Any) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        result = results[0]
        prompt_map = self.config.prompt_map
        label_map = self.config.label_map
        detections: list[Detection] = []
        boxes = result.boxes
        for xyxy, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            model_name = str(result.names[int(class_id.item())])
            vision_class = prompt_map.get(model_name) if self.open_vocabulary else label_map.get(model_name)
            if vision_class is None:
                continue
            coords = tuple(int(round(value)) for value in xyxy.detach().cpu().tolist())
            detections.append(
                Detection(
                    label=vision_class.label,
                    prompt=model_name,
                    confidence=float(confidence.item()),
                    box_xyxy=coords,
                )
            )
        return detections
