# ChatBot For Support Portal

PersonalRAG is a lightweight Retrieval-Augmented Generation (RAG) chatbot starter that ingests your documents, stores semantic embeddings locally, and serves an interactive CLI for question answering. Use it as-is for internal knowledge assistants or embed its modules inside your own automation.

---

## Prerequisites
- Python 3.10+ with `pip`.
- `make` (optional, used for shortcuts below).
- Disk space for sentence-transformer weights and the Chroma vector store (hundreds of MB).
- Hugging Face access for the reasoning model (defaults to `google/flan-t5-small`).

---

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
> The first run downloads the transformer model and persists it under `models/`. Leave `models/` cached if you plan to reuse the chatbot offline.

---

## Quick Start
```bash
# Ingest docs (optional) and launch the chat CLI
python main.py --source docs --reset --greeting "Welcome to PersonalRAG!"
```
During the session you can use slash commands (e.g., `/ingest`, `/stats`, `/topk 8`, `/greeting Hello`).

---

## Using the Makefile
This repo includes a convenience `Makefile`. Common targets:

```bash
make install             # create .venv and install dependencies
make run ARGS="--source docs --reset"
make ingest SOURCE=~/Downloads/policies RESET=1
make clean-store         # wipe the Chroma DB
```
> Pass additional CLI flags through `ARGS="..."`. The `ingest` target expects a `SOURCE` path and optional `RESET=1` or `RECURSIVE=0`.

---

## CLI Reference (`main.py`)
| Flag | Description |
| --- | --- |
| `--source <path>` | Ingest a file or directory before the chat session starts. |
| `--reset` | Clear the Chroma store before ingesting. |
| `--no-recursive` | Disable recursive directory walks during ingestion. |
| `--default-source <path>` | Override the fallback docs directory. |
| `--top-k <int>` | Number of contexts sent to the LLM per answer (default 5). |
| `--greeting "<text>"` | Custom welcome message. |

Slash commands available inside the chat:
- `/ingest <path> [--reset] [--no-recursive]`
- `/stats`
- `/topk <n>`
- `/greeting <text>`
- `/exit`

---

## Module Guide
| Path | Responsibility |
| --- | --- |
| `main.py` | CLI harness: argument parsing, command handling, chat loop orchestration. |
| `utils/chatbot.py` | `RAGChatbot` class for ingestion, session tracking, answering, and stats. |
| `utils/loader.py` | Multi-format document loader (TXT/MD/PDF/DOCX/JSON/CSV) with recursive traversal. |
| `utils/chunking.py` | Sentence-aware chunking via NLTK plus helper iterators for bulk documents. |
| `utils/embeddings.py` | Embedding pipelines (SentenceTransformers), Chroma persistence, semantic search helpers. |
| `utils/rerankers.py` | Lightweight lexical re-ranking that mixes cosine scores with keyword overlap. |
| `utils/reasoning.py` | Wrapper around Hugging Face transformers for grounded generation over retrieved contexts. |
| `models/` | Cached transformer weights downloaded the first time you run the chatbot. |
| `store/` | On-disk Chroma vector store containing embeddings. |
| `docs/` | Default knowledge assets. Update this folder or point the CLI to another directory. |

---

## Reusing in Your Project
1. **Install as a module:** Add this repo to your workspace, `pip install -r requirements.txt`, and import `RAGChatbot`.
   ```python
   from utils.chatbot import RAGChatbot
   bot = RAGChatbot(default_source="my_docs", greeting="Hi from Acme!")
   bot.create_chat(reset=True)
   print(bot.answer("What is our refund policy?"))
   ```
2. **Swap the retriever or LLM:** Replace `utils/embeddings.py` with your preferred embedding backend or point `reasoning()` to a hosted model by changing the default `model_id`.
3. **Embed inside services:** Use `bot.answer()` inside FastAPI/Slack bots. Persist the Chroma store (`store/`) wherever your service runs.
4. **Extend ingestion:** Add new readers in `utils/loader.py` for custom file formats (e.g., HTML) and call `bot.ingest(source=..., recursive=False)`.

---

## Response Providers
- `structured` (default): Picks the best matching lines from retrieved docs and crafts a friendly sentence or two—great for predictable answers like phone numbers or policy snippets.
- `local`: Uses the bundled Hugging Face model (`google/flan-t5-small`) with strict formatting guards; set with `export RAG_REASONING_PROVIDER=local`.
- `openai`: Calls ChatGPT via the OpenAI API (see below). Handy when you want more natural phrasing or longer answers.

Switch providers any time with `RAG_REASONING_PROVIDER=<structured|local|openai>`.

---

## Switch to OpenAI ChatGPT Responses
Prefer ChatGPT-style answers over the local Hugging Face model? Toggle the reasoning backend without touching code:

```bash
export OPENAI_API_KEY="sk-..."               # required
export RAG_REASONING_PROVIDER=openai         # tells the bot to use OpenAI
export RAG_OPENAI_MODEL=gpt-4o-mini          # optional override
python main.py --source docs
```

The fallback remains the bundled `google/flan-t5-small` weights, so you can revert by unsetting `RAG_REASONING_PROVIDER` (or setting it to `local`).

---

## Operational Tips
- Keep `docs/` structured by topic to make the retrieval context cleaner.
- Regenerate embeddings (`--reset`) whenever you remove outdated documents; otherwise stale chunks remain in the store.
- Track model downloads (`models/`) in `.gitignore` to avoid committing large files.
- For CI or headless deployments, set `HF_HUB_OFFLINE=1` after the initial download to prevent repeated pulls.
- Tune prompt size via `RAG_FACT_CHARS_PER_DOC` and `RAG_FACT_TOTAL_CHARS` if you need shorter or longer context windows for the reasoning step.

---

## Need More?
- See `docs/SupportDocs.MD` for a fuller support playbook.
- Add additional automation by extending the `Makefile` or wiring `RAGChatbot` into your preferred orchestration framework.
