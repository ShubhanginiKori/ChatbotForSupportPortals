from __future__ import annotations

import chromadb

from pathlib import Path
from typing import Iterable
from uuid import uuid4
from sentence_transformers import SentenceTransformer

from utils.chunking import chunk_documents, chunk_text

_MODEL_NAME = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "personal_rag_documents"
_CHROMA_DIR = Path(__file__).resolve().parent.parent / "store"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(_MODEL_NAME)
_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

_collection = _client.get_or_create_collection(
    name=_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def create_embeddings(text: str | Iterable[str]) -> list[tuple[str, list[float]]]:
    if isinstance(text, str):
        docs = chunk_text(text)
    else:
        try:
            iter(text)  # type: ignore[arg-type]
            docs = list(chunk_documents(text))
        except TypeError:
            docs = list(chunk_documents([text]))  # type: ignore[arg-type]

    if not docs:
        return []

    encoded = model.encode(
        docs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [(doc, vector.tolist()) for doc, vector in zip(docs, encoded)]


def store_embeddings(embedding: list[tuple[str, list[float]]] | tuple[str, list[float]]):
    if not embedding:
        return []

    pairs: list[tuple[str, list[float]]]
    if isinstance(embedding, list):
        pairs = embedding
    else:
        pairs = [embedding]

    documents: list[str] = []
    embeddings: list[list[float]] = []
    ids: list[str] = []

    for text, vector in pairs:
        if not text:
            continue
        documents.append(text)
        embeddings.append(list(vector))
        ids.append(str(uuid4()))

    if not documents:
        return []

    _collection.add(ids=ids, documents=documents, embeddings=embeddings)
    return ids


def match_query(query: str, *, top_k: int = 10) -> list[dict[str, object]]:
    query = query.strip()
    if not query:
        return []

    if _collection.count() == 0:
        return []

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    top_k = max(1, min(top_k, _collection.count()))
    response = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    documents = response.get("documents", [[]])[0]
    distances = response.get("distances", [[]])[0]
    ids = response.get("ids", [[]])[0]

    results: list[dict[str, object]] = []
    for doc_id, document, distance in zip(ids, documents, distances):
        score = 1.0 - float(distance) if distance is not None else 0.0
        if score < 0.0:
            score = 0.0
        results.append({"id": doc_id, "text": document, "score": score})
    return results


def reset_store() -> None:
    _collection.delete(where={})


def store_size() -> int:
    return _collection.count()


if __name__ == "__main__":
    from loader import load_documents

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    documents_ = load_documents(source=docs_dir)

    for doc_ in documents_:
        embeddings_ = create_embeddings([doc_])
        if embeddings_:
            store_embeddings(embeddings_)

    result_sets = match_query("Where are you from ?")
    print(result_sets)
