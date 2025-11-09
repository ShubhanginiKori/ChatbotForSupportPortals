from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

# Backend selection
_DEFAULT_PROVIDER = os.getenv("RAG_REASONING_PROVIDER", "structured").lower()
_DEFAULT_OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")

# Conservative limits so small models don't explode on context
_MAX_FACT_CHARS_PER_DOC = int(os.getenv("RAG_FACT_CHARS_PER_DOC", "400"))
_MAX_FACT_TOTAL_CHARS = int(os.getenv("RAG_FACT_TOTAL_CHARS", "1800"))

_SYSTEM_PROMPT = (
    "You are a grounded support assistant. "
    "You must answer ONLY using the provided facts. "
    "If the information is missing, reply exactly: Not stated in any of documents."
)


def reasoning(
    documents: Iterable[str],
    question: str,
    *,
    model_id: str = "google/flan-t5-small",
    models_dir: str = "../models",
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    provider: str | None = None,
    openai_model: str | None = None,
) -> str:
    """
    Generate a grounded answer from retrieved document chunks.

    Modes:
        structured (default): deterministic summarizer that picks the most relevant
            lines and rewrites them into a conversational response.
        local: call a local Hugging Face model (same behaviour as before, with
            the single-sentence / 20-word guard rails).
        openai: use ChatGPT-style completions via the OpenAI API.
    """

    contexts = [doc for doc in documents if doc and str(doc).strip()]
    if not contexts:
        return "Not stated in any of documents."

    backend = (provider or _DEFAULT_PROVIDER).lower()

    if backend in {"structured", "simple", "rule"}:
        return _structured_response(question, contexts)

    prompt = _build_prompt(contexts, question)

    if backend == "openai":
        raw = _run_openai(
            prompt=prompt,
            model_name=openai_model or _DEFAULT_OPENAI_MODEL,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        raw = _run_local(
            prompt=prompt,
            model_id=model_id,
            models_dir=models_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return _finalize_answer(raw)


def _build_prompt(documents: Iterable[str], question: str) -> str:
    facts = _prepare_facts(documents)

    return f"""
        Use only the statements in FACTS to answer QUESTION.
        
        Rules:
        - Answer in ONE short, friendly sentence.
        - Maximum 20 words.
        - Plain text only: no lists, no tables, no markdown, no code.
        - Include only details directly supported by FACTS.
        - If the requested information is missing, reply exactly: Not stated in any of documents.
        
        FACTS:
        {facts or 'No supporting facts were provided.'}
        
        QUESTION:
        {question}
        
        Final answer (one friendly sentence, max 20 words):
    """.strip()


def _prepare_facts(documents: Iterable[str]) -> str:
    """
    Normalize and trim documents so they fit small-model context.
    """
    snippets: list[str] = []
    remaining = _MAX_FACT_TOTAL_CHARS

    for doc in documents:
        if remaining <= 0:
            break

        text = (doc or "").strip()
        if not text:
            continue

        snippet = text[:_MAX_FACT_CHARS_PER_DOC].strip()
        if not snippet:
            continue

        snippet = " ".join(snippet.split())
        if not snippet:
            continue

        use_len = min(len(snippet), remaining)
        snippets.append(f"- {snippet[:use_len]}")
        remaining -= use_len

    return "\n".join(snippets)


def _structured_response(question: str, contexts: list[str]) -> str:
    question_clean = question.strip()
    question_lower = question_clean.lower()
    question_tokens = _tokenize(question_clean)
    candidates: list[tuple[float, str]] = []

    for ctx in contexts:
        for line in _extract_lines(ctx):
            normalized = _normalize_line(line)
            if not normalized:
                continue
            tokens = _tokenize(normalized)
            score = _lexical_similarity(question_tokens, tokens)
            if "number" in question_lower and any(ch.isdigit() for ch in normalized):
                score += 0.4
            if "helpdesk" in question_lower and "helpdesk" in normalized.lower():
                score += 0.25
            for keyword in ("amer", "emea", "apac", "asia", "europe", "america"):
                if keyword in question_lower and keyword in normalized.lower():
                    score += 0.25
            if any(char.isdigit() for char in normalized):
                score += 0.05
            if score <= 0 and len(tokens) < 3:
                continue
            candidates.append((score, normalized))

    if not candidates:
        return "Not stated in any of documents."

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_lines: list[str] = []
    seen: set[str] = set()
    for _, line in candidates:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        best_lines.append(line)
        if len(best_lines) == 3:
            break

    intro = _build_intro(question_clean)
    main = _line_to_sentence(best_lines[0])
    if len(best_lines) == 1:
        return f"{intro} {main}"

    bullets = "\n".join(f"- {_line_to_sentence(line)}" for line in best_lines[1:])
    return f"{intro} {main}\n{bullets}"


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _lexical_similarity(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    shared = sum((query_counts & doc_counts).values())
    if shared == 0:
        return 0.0
    coverage = shared / len(query_tokens)
    unique_query = set(query_tokens)
    unique_doc = set(doc_tokens)
    union = len(unique_query | unique_doc)
    intersection = len(unique_query & unique_doc)
    jaccard = intersection / union if union else 0.0
    return (0.7 * coverage) + (0.3 * jaccard)


def _extract_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            lines.append(stripped)
            continue
        parts = re.split(r"(?<=[.!?])\s+|\s{2,}", stripped)
        for part in parts:
            part = part.strip(" -*•")
            if len(part) >= 3:
                lines.append(part)
    return lines


def _normalize_line(line: str) -> str:
    cleaned = line.strip(" -*•")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _line_to_sentence(line: str) -> str:
    if "|" in line:
        cells = [cell.strip(" `") for cell in line.split("|") if cell.strip()]
        if len(cells) >= 4:
            region, hours, phone, contact = cells[:4]
            contact = contact.replace("`", "").replace("/", " or ").strip()
            return (
                f"{region} operates {hours} and you can reach them at {phone} via {contact}."
            )
        if len(cells) == 3:
            a, b, c = cells
            return f"{a} runs {b} and can be reached at {c}."
        if len(cells) == 2:
            a, b = cells
            return f"{a}: {b}."
    return line


def _build_intro(question: str) -> str:
    topic = question.rstrip("?.!").strip()
    if not topic:
        return "Here’s what I found:"
    if topic.lower().startswith("what"):
        topic = topic[4:].strip()
    return f"You asked about {topic or 'that request'}. Here’s what I found:"


def _run_local(
    *,
    prompt: str,
    model_id: str,
    models_dir: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Run a local HF model (free). Works with encoder-decoder or causal models.
    """
    name = model_id.split("/")[-1]
    path = Path(models_dir) / name
    path.mkdir(parents=True, exist_ok=True)

    if not (path / "config.json").exists():
        if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true"}:
            raise RuntimeError(
                f"Model {model_id} not found at {path}. "
                f"Disable HF_HUB_OFFLINE once to download weights."
            )
        snapshot_download(
            model_id,
            local_dir=str(path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)

    if getattr(cfg, "is_encoder_decoder", False):
        mdl = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
        pipe = pipeline(
            "text2text-generation",
            model=mdl,
            tokenizer=tok,
        )
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            truncation=True,
        )[0]["generated_text"]
    else:
        mdl = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
        pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
        )
        res = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            eos_token_id=pipe.tokenizer.eos_token_id,
            truncation=True,
        )[0]["generated_text"]
        out = res[len(prompt) :] if res.startswith(prompt) else res

    return out.strip()


def _run_openai(
    *,
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Optional: use OpenAI as reasoning backend.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set to use the OpenAI backend."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install 'openai' package to enable the OpenAI backend."
        ) from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    if not response.choices:
        return ""

    return (response.choices[0].message.content or "").strip()


def _finalize_answer(raw: str) -> str:
    """
    Normalize model output to:
    - single plain sentence
    - friendly tone
    - <= 20 words
    - or exact 'Not stated in any of documents.'
    """
    if not raw:
        return "Not stated in any of documents."

    text = raw.strip()

    # If model echoed the whole prompt, keep only after the last 'Final answer'
    lower = text.lower()
    if "final answer" in lower:
        idx = lower.rfind("final answer")
        text = text[idx + len("final answer") :].strip()

    # Remove labels / markdown / bullets
    text = re.sub(r"\b(facts|question)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^[\-\*\u2022]\s*", "", text)  # leading bullets
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    # Check grounded "not stated" response
    if "not stated in any of documents" in text.lower():
        return "Not stated in any of documents."

    # Strip any markdown artifacts
    text = re.sub(r"[*`_]+", "", text)

    # Make sure it's one sentence: cut at first strong terminator if model rambled
    m = re.search(r"(.+?[.!?])\s", text)
    if m:
        text = m.group(1)

    # Enforce 20-word limit
    words = text.split()
    if len(words) > 20:
        words = words[:20]
        text = " ".join(words)

    # Light friendly tweak: add soft prefix if it's too bare and we have room.
    # (No question-specific or domain-specific hardcoding.)
    if len(words) <= 16 and not re.match(r"(?i)^(here|hi|hello|sure|you|the|this|that|it)\b", text):
        prefix = "Here's what I found:"
        prefix_words = prefix.split()
        if len(prefix_words) + len(words) <= 20:
            text = f"{prefix} {' '.join(words)}"
            words = text.split()

    # Clean trailing junk, ensure it feels like a sentence
    text = text.strip(" ,;:-")
    if not text.endswith((".", "!", "?")):
        text += "."

    # Final guard
    if not text:
        return "Not stated in any of documents."

    return text




if __name__ == "__main__":
    # Example documents (can be any text extracted from MD, docx, etc.)
    documents = [
        "Sivaram is a software engineer at McKinsey and loves AI systems.",
        "He enjoys travelling on weekends and exploring new technologies.",
        "Americas helpdesk operates 24/7 and can be reached at +1-415-555-0110.",
    ]

    # Example questions
    questions = [
        "Who is Sivaram?",
        "What is Americas helpdesk number?",
        "When can I reach Americas helpdesk?",
        "What is Sivaram's favorite hobby?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = reasoning(documents, q)
        print(f"A: {answer}")
