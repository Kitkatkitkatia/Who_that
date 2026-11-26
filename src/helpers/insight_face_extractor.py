import numpy as np

class InsightFaceExtractor:
    """Optional InsightFace backend (non-commercial research weights)."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_thresh: float = 0.4,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = -1,
    ):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)

    def get_primary_embedding(self, img_bgr: np.ndarray) -> np.ndarray | None:
        faces = self.get_all_embeddings(img_bgr)
        if not faces:
            return None
        primary = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        return primary["embedding"]

    def get_all_embeddings(self, img_bgr: np.ndarray) -> list[dict]:
        return [
            {
                "bbox": [round(float(v), 1) for v in face.bbox],
                "embedding": np.asarray(face.embedding, dtype=np.float32),
            }
            for face in self.app.get(img_bgr)
        ]