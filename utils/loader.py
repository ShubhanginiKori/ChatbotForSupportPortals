from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # type: ignore

try:
    from docx import Document  # type: ignore
except ImportError:
    Document = None  # type: ignore


Readable = Callable[[Path], str]
DEFAULT_SUFFIXES: tuple[str, ...] = (
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".json",
    ".csv",
)


def load_documents(
    source: str | Path,
    *,
    recursive: bool = True,
    allowed_suffixes: Iterable[str] | None = None,
) -> list[str]:
    """
    Collect textual content from documents for embedding generation.

    Args:
        source: File or directory containing training material.
        recursive: When ``True`` traverse sub-directories.
        allowed_suffixes: Override the default set of supported extensions.

    Returns:
        A list of document contents as strings.
    """

    root = Path(source).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"No such file or directory: {root}")

    suffix_filter = tuple(
        s.lower() if s.startswith(".") else f".{s.lower()}"
        for s in (allowed_suffixes or DEFAULT_SUFFIXES)
    )

    readers: dict[str, Readable] = {
        ".txt": _read_text,
        ".md": _read_text,
        ".json": _read_json,
        ".csv": _read_csv,
        ".pdf": _read_pdf,
        ".docx": _read_docx,
    }

    documents: list[str] = []
    for path in _iter_paths(root, recursive):
        suffix = path.suffix.lower()
        if suffix not in suffix_filter:
            continue
        reader = readers.get(suffix)
        if not reader:
            continue
        text = reader(path)
        if text:
            documents.append(text)
    return documents


def _iter_paths(root: Path, recursive: bool) -> Iterator[Path]:
    if root.is_file():
        yield root
        return

    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and not path.name.startswith("."):
            yield path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(_flatten_json(data))


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                rows.append(" ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(rows)


def _read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise ImportError(
            "PyPDF2 is required to read PDF files. Install it via `pip install PyPDF2`."
        )
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        extracted = extracted.strip()
        if extracted:
            texts.append(extracted)
    return "\n".join(texts)


def _read_docx(path: Path) -> str:
    if Document is None:
        raise ImportError(
            "python-docx is required to read DOCX files. Install it via `pip install python-docx`."
        )
    document = Document(str(path))
    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]
    return "\n".join(paragraphs)


def _flatten_json(data: object) -> Iterator[str]:
    stack: list[object] = [data]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(reversed(value.values()))
        elif isinstance(value, (list, tuple, set)):
            stack.extend(reversed(list(value)))
        elif value is not None:
            yield str(value)


if __name__ == "__main__":
    docs = load_documents(source="../docs")
    print(docs)
