from __future__ import annotations

import nltk
from typing import Iterable, Iterator, Sequence

_NLTK_MODEL_READY = False


def _ensure_nltk_ready() -> None:
    global _NLTK_MODEL_READY
    if _NLTK_MODEL_READY:
        return

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    _NLTK_MODEL_READY = True


def chunk_text(text: str) -> list[str]:
    """
    Produce sentence-aware (semantic) chunks using NLTK's sentence tokenizer.
    """
    _ensure_nltk_ready()

    sentences = nltk.sent_tokenize(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_documents(docs: Sequence[str] | Iterable[str]) -> Iterator[str]:
    for doc in docs:
        for chunk in chunk_text(str(doc)):
            yield chunk


if __name__ == "__main__":
    sentences_txt = (
        "Artificial Intelligence is transforming the world.\n"
        "It powers chatbots, recommendation systems, and autonomous cars.\n"
        "Machine Learning is a subset of AI. Deep Learning uses neural networks to learn from data On the other hand. AI ethics focuses on fairness and transparency. Regulations are being discussed globally."
    )

    for chunk_ in chunk_text(sentences_txt):
        print(chunk_)
