import argparse
import json
import os
import sys

import ebooklib
from ebooklib import epub
from pypdf import PdfReader

from vernadskiy_boltalka.paths import project_root


def _chunk_with_unstructured(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    try:
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.text import partition_text

        elements = partition_text(text=text)
        chunks = chunk_by_title(elements, max_characters=chunk_size, overlap=overlap)
        return [str(c).strip() for c in chunks if str(c).strip()]
    except ImportError:
        return []


def _chunk_simple(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ", chunk_size - overlap)
            if last_space > chunk_size // 2:
                end = start + last_space + 1
                chunk = text[start:end]
        c = chunk.strip()
        if c:
            chunks.append(c)
        start = end
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    use_unstructured: bool = False,
) -> list[str]:
    if use_unstructured:
        out = _chunk_with_unstructured(text, chunk_size, overlap)
        if out:
            return out
    return _chunk_simple(text, chunk_size, overlap)


class Processor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_unstructured: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_unstructured = use_unstructured

    def load_from_dir(self, data_dir: str) -> list[tuple[str, str]]:
        results = []
        if not os.path.isdir(data_dir):
            return results

        for name in sorted(os.listdir(data_dir)):
            full = os.path.join(data_dir, name)
            ext = os.path.splitext(name)[1].lower()
            if ext == ".pdf":
                t = self._extract_pdf(full)
            elif ext == ".epub":
                t = self._extract_epub(full)
            elif ext == ".docx":
                t = self._extract_docx(full)
            else:
                continue
            text = (t or "").strip()
            if text:
                results.append((name, text))
        return results

    def _extract_pdf(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            print(f"  PDF {os.path.basename(path)}: ошибка — {e}")
            return ""

    def _extract_epub(self, path: str) -> str:
        try:
            book = epub.read_epub(path)
            parts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    parts.append(soup.get_text(separator="\n"))
            return "\n\n".join(parts)
        except Exception:
            return ""

    def _extract_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    def chunk_texts(self, texts: list[tuple[str, str]]) -> list[dict]:
        chunks = []
        for source, text in texts:
            parts = chunk_text(
                text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                use_unstructured=self.use_unstructured,
            )
            for i, part in enumerate(parts):
                chunks.append({
                    "text": part,
                    "meta": {"source": source, "chunk_idx": i},
                })
        return chunks

    def process_dir(self, data_dir: str) -> list[dict]:
        texts = self.load_from_dir(data_dir)
        return self.chunk_texts(texts)


def load_texts_from_dir(data_dir: str) -> list[tuple[str, str]]:
    return Processor().load_from_dir(data_dir)


Preprocessor = Processor

PREPROCESSED_PATH = os.path.join(project_root(), "data", "preprocessed_chunks.json")


def save_chunks(chunks: list[dict], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {path} ({len(chunks)} чанков)")


def load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(data_dir: str, out_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
    processor = Processor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_dir(data_dir)
    if not chunks:
        sys.exit(1)
    save_chunks(chunks, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предобработка текстов → чанки → JSON")
    parser.add_argument("--data-dir", type=str, default=None, help="Папка с PDF/EPUB/DOCX")
    parser.add_argument("--out", type=str, default=None, help="Выходной JSON (по умолч. data/preprocessed_chunks.json)")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    root = project_root()
    data_dir = args.data_dir or os.path.join(root, "vernadskiy_data")
    out_path = args.out or PREPROCESSED_PATH
    run(data_dir=data_dir, out_path=out_path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
