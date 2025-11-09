from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

from utils.embeddings import (
    create_embeddings,
    match_query,
    reset_store,
    store_embeddings,
    store_size,
)
from utils.loader import load_documents
from utils.reasoning import reasoning
from utils.rerankers import rerank_results

_DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@dataclass(slots=True)
class IngestStats:
    documents: int = 0
    chunks: int = 0
    store_size: int = 0


class RAGChatbot:
    def __init__(
        self,
        *,
        default_source: str | Path | None = None,
        greeting: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        self._default_source = (
            Path(default_source).expanduser().resolve()
            if default_source
            else _DEFAULT_DOCS_DIR
        )
        self._top_k = max(1, top_k)
        self._chat_id: Optional[str] = None
        self._stats = IngestStats(store_size=store_size())
        self.greeting = greeting or "Hi! Ask me something about your documents."

    @property
    def chat_id(self) -> Optional[str]:
        return self._chat_id

    @property
    def stats(self) -> IngestStats:
        return self._stats

    def create_chat(
        self,
        *,
        source: str | Path | None = None,
        reset: bool = False,
        recursive: bool = True,
    ) -> str:
        if reset:
            reset_store()

        path = Path(source).expanduser().resolve() if source else self._default_source
        documents = load_documents(path, recursive=recursive) if path.exists() else []

        chunk_count = 0
        for document in documents:
            embeddings = create_embeddings(document)
            if not embeddings:
                continue
            stored_ids = store_embeddings(embeddings)
            chunk_count += len(stored_ids)

        self._stats = IngestStats(
            documents=len(documents),
            chunks=chunk_count,
            store_size=store_size(),
        )
        self._chat_id = str(uuid4())
        return self._chat_id

    def chat_session(
        self, current_question: str, *, top_k: Optional[int] = None
    ) -> str:
        question = current_question.strip()
        if not question:
            return ""

        limit = max(1, top_k or self._top_k)
        candidates = match_query(question, top_k=limit * 2)
        reranked = rerank_results(question, candidates, top_k=limit)
        contexts = [item["text"].strip() for item in reranked if item.get("text")]

        if not contexts:
            return "I could not find matching information in the document store."

        return reasoning(contexts, question)

    def ingest(
        self,
        *,
        source: str | Path,
        reset: bool = False,
        recursive: bool = True,
    ) -> dict[str, int | str]:
        chat_id = self.create_chat(
            source=source,
            reset=reset,
            recursive=recursive,
        )
        return {
            "chat_id": chat_id,
            "documents": self._stats.documents,
            "chunks": self._stats.chunks,
            "store_size": self._stats.store_size,
        }

    def answer(self, question: str, *, top_k: Optional[int] = None) -> str:
        return self.chat_session(question, top_k=top_k)


def run_chat_loop(bot: "RAGChatbot") -> None:
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: See you later!")
            break

        reply = bot.answer(user_input)
        if not reply:
            reply = "I could not understand that. Could you rephrase?"
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    bot = RAGChatbot(greeting="hi im ai assistant to guide you")
    bot.create_chat()

    print(bot.greeting)
    print(
        f"Loaded {bot.stats.documents} documents into {bot.stats.chunks} chunks "
        f"(store size {bot.stats.store_size}). Type 'exit' to quit."
    )

    run_chat_loop(bot)
