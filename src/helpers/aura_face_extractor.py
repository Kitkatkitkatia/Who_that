import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
AURAFACE_URL = "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx"


def _fetch(url: str, path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return str(path)


class AuraFaceExtractor:
    """Default AuraFace backend (Apache-2.0/MIT weights)."""

    def __init__(
        self,
        cache_dir: Path = Path.home() / ".cache" / "face-embeddings",
        det_thresh: float = 0.4,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = -1,
    ):
        yunet = _fetch(YUNET_URL, cache_dir / "face_detection_yunet_2023mar.onnx")
        auraface = _fetch(AURAFACE_URL, cache_dir / "auraface_glintr100.onnx")

        self.max_edge = max(det_size)
        self.detector = cv2.FaceDetectorYN.create(yunet, "", (320, 320), det_thresh)

        providers = (
            [("CUDAExecutionProvider", {"device_id": ctx_id}), "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(auraface, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img_bgr.shape[:2]
        scale = min(1.0, self.max_edge / max(h, w)) if self.max_edge > 0 else 1.0
        det_img = (
            cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else img_bgr
        )

        self.detector.setInputSize((det_img.shape[1], det_img.shape[0]))
        _, faces = self.detector.detect(det_img)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32), scale
        return faces, scale

    def _embed(self, img_bgr: np.ndarray, row: np.ndarray, scale: float) -> np.ndarray | None:
        kps = (row[4:14] / scale).reshape(5, 2).astype(np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
        if matrix is None:
            return None

        aligned = cv2.warpAffine(img_bgr, matrix, (112, 112))
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), True)
        out = np.asarray(self.session.run(None, {self.input_name: blob})[0], dtype=np.float32)
        return out[0]

    def get_primary_embedding(self, img_bgr: np.ndarray) -> np.ndarray | None:
        faces, scale = self._detect(img_bgr)
        if len(faces) == 0:
            return None
        primary = max(faces, key=lambda r: r[2] * r[3])
        return self._embed(img_bgr, primary, scale)

    def get_all_embeddings(self, img_bgr: np.ndarray) -> list[dict]:
        faces, scale = self._detect(img_bgr)
        results = []
        for row in faces:
            embedding = self._embed(img_bgr, row, scale)
            if embedding is None:
                continue
            x, y, w, h = row[:4] / scale
            results.append(
                {
                    "bbox": [round(float(x), 1), round(float(y), 1), round(float(x + w), 1), round(float(y + h), 1)],
                    "embedding": embedding,
                }
            )
        return results