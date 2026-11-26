import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def get_face_embeddings(extractor, image_folder_path: str) -> dict[str, np.ndarray]:
    embeddings_dict = {}
    for folder in sorted(Path(image_folder_path).iterdir()):
        if folder.is_dir():
            embeddings = []
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    img = cv2.imread(str(image_path))
                    if img is None:
                        print(f"Could not read image {image_path}")
                        continue
                    emb = extractor.get_primary_embedding(img)
                    if emb is not None:
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            embeddings.append(emb / norm)
                    else:
                        print(f"No face detected in {image_path}")
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                avg_norm = np.linalg.norm(avg_embedding)
                if avg_norm > 0:
                    embeddings_dict[folder.name] = (avg_embedding / avg_norm).astype(np.float32)
            else:
                print(f"No valid embeddings found in folder {folder.name}")
    return embeddings_dict


def main():
    parser = argparse.ArgumentParser(description="Generate face embeddings for images in a folder")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="path to the folder containing images")
    parser.add_argument("--output_file", type=str, required=True,
                        help="path to the output file to save embeddings (.npz format)")
    parser.add_argument("--backend", type=str, choices=["auraface", "insightface"], default="auraface",
                        help="recognition backend (default: 'auraface' [Apache-2.0/MIT]; "
                             "'insightface' downloads non-commercial research weights)")
    parser.add_argument("--model_name", type=str, default="buffalo_l",
                        help="InsightFace model pack name when --backend insightface is used")
    parser.add_argument("--det_thresh", type=float, default=0.4,
                        help="detection threshold for face detection")
    parser.add_argument("--det_size", type=int, nargs=2, default=(640, 640),
                        help="detection size for face detection")
    parser.add_argument("--ctx_id", type=int, default=-1,
                        help="context id for the model (-1 for CPU, >=0 for GPU)")
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / ".cache" / "face-embeddings",
                        help="directory to cache downloaded ONNX weights")

    args = parser.parse_args()
    det_size = tuple(args.det_size)

    if args.backend == "auraface":
        from src.helpers.aura_face_extractor import AuraFaceExtractor
        extractor = AuraFaceExtractor(
            cache_dir=args.cache_dir,
            det_thresh=args.det_thresh,
            det_size=det_size,
            ctx_id=args.ctx_id,
        )
    else:
        from src.helpers.insight_face_extractor import InsightFaceExtractor
        extractor = InsightFaceExtractor(
            model_name=args.model_name,
            det_thresh=args.det_thresh,
            det_size=det_size,
            ctx_id=args.ctx_id,
        )

    num_subfolders = sum(1 for p in Path(args.input_folder).iterdir() if p.is_dir())
    print(f"Processing {num_subfolders} folders in {args.input_folder} using backend '{args.backend}'")

    embeddings_dict = get_face_embeddings(extractor, args.input_folder)

    output_path = Path(args.output_file).with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **embeddings_dict)  # type: ignore

    print(f"Saved centroid embeddings for {len(embeddings_dict)} identities to {output_path}")

if __name__ == "__main__":
    main()