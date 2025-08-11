import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Simple configuration
INPUT_DIR = Path("markdown_files_crawler")
INDEX_DIR = Path("faiss_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def discover_markdown_paths(input_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for root, _dirs, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".md"):
                paths.append(Path(root) / name)
    return sorted(paths)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[str, int, int]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: List[Tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((chunk, start, end))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(input_dir: Path) -> Tuple[List[str], List[dict]]:
    texts: List[str] = []
    metadata: List[dict] = []
    for file_path in tqdm(discover_markdown_paths(input_dir), desc="Reading and chunking"):
        try:
            content = read_text(file_path)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[WARN] Skipping {file_path}: {exc}")
            continue

        rel = str(file_path.relative_to(input_dir))
        for idx, (chunk, start, end) in enumerate(chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)):
            texts.append(chunk)
            metadata.append(
                {
                    "source": rel,
                    "abs_source": str(file_path),
                    "chunk_index": idx,
                    "start_char": start,
                    "end_char": end,
                }
            )
    return texts, metadata


def embed(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32, copy=False)


def save_index(vectors: np.ndarray, metadata: List[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    with (INDEX_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    with (INDEX_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "embedding_dimension": dim,
                "num_vectors": int(vectors.shape[0]),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    texts, metadata = build_chunks(INPUT_DIR)
    if not texts:
        raise SystemExit(f"No markdown files found in '{INPUT_DIR}'.")
    vectors = embed(texts)
    save_index(vectors, metadata)
    print(f"Indexed {len(texts)} chunks from '{INPUT_DIR}' into '{INDEX_DIR}'.")


if __name__ == "__main__":
    main()


