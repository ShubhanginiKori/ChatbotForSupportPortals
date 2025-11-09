from __future__ import annotations

import argparse
import shlex
import sys

from utils.chatbot import RAGChatbot


def build_parser() -> argparse.ArgumentParser:
    """Create a simple argument parser for configuring the chatbot CLI."""
    parser = argparse.ArgumentParser(
        description="Interactive PersonalRAG chatbot.",
    )
    parser.add_argument(
        "--source",
        help="Optional file or directory to ingest before starting the chat.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store before the initial ingestion.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive traversal when ingesting directories.",
    )
    parser.add_argument(
        "--default-source",
        help="Override the default source directory for the chatbot.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to consider per answer (default: 5).",
    )
    parser.add_argument(
        "--greeting",
        help="Custom greeting shown when the chat session starts.",
    )
    return parser


def _print_startup(bot: RAGChatbot) -> None:
    stats = bot.stats
    print(bot.greeting)
    print(
        f"Assistant: Loaded {stats.documents} documents into {stats.chunks} chunks "
        f"(store size {stats.store_size})."
    )
    print("Assistant: Type your message or '/help' to view available commands.")


def _handle_command(
    bot: RAGChatbot, command: str, top_k: int, recursive_default: bool
) -> tuple[int, bool, bool]:
    tokens = shlex.split(command)
    if not tokens:
        return top_k, recursive_default, False

    name = tokens[0].lower()

    if name in {"exit", "quit"}:
        return top_k, recursive_default, True

    if name == "help":
        print(
            "Assistant: Available commands:\n"
            "  /ingest <path> [--reset] [--no-recursive]  Ingest new documents.\n"
            "  /stats                                     Show document store stats.\n"
            "  /topk <n>                                  Adjust retrieval depth.\n"
            "  /greeting <text>                           Update the assistant greeting.\n"
            "  /exit                                      Leave the chat."
        )
        return top_k, recursive_default, False

    if name == "ingest":
        if len(tokens) == 1:
            print("Assistant: Usage: /ingest <path> [--reset] [--no-recursive]")
            return top_k, recursive_default, False

        source = None
        reset = False
        recursive = recursive_default

        for token in tokens[1:]:
            if token == "--reset":
                reset = True
            elif token in {"--no-recursive", "--norecursive"}:
                recursive = False
            elif token == "--recursive":
                recursive = True
            elif token.startswith("--"):
                print(f"Assistant: Unknown option '{token}'.")
                return top_k, recursive_default, False
            elif source is None:
                source = token
            else:
                print("Assistant: Only one source path can be provided.")
                return top_k, recursive_default, False

        if source is None:
            print("Assistant: Please provide a path to ingest.")
            return top_k, recursive_default, False

        stats = bot.ingest(
            source=source,
            reset=reset,
            recursive=recursive,
        )
        recursive_default = recursive
        print(
            f"Assistant: Loaded {stats['documents']} documents into {stats['chunks']} chunks "
            f"(store size {stats['store_size']})."
        )
        return top_k, recursive_default, False

    if name == "stats":
        stats = bot.stats
        print(
            f"Assistant: Documents {stats.documents}, chunks {stats.chunks}, "
            f"store size {stats.store_size}."
        )
        return top_k, recursive_default, False

    if name == "topk":
        if len(tokens) < 2:
            print("Assistant: Usage: /topk <positive integer>")
            return top_k, recursive_default, False

        try:
            value = int(tokens[1])
            if value <= 0:
                raise ValueError
        except ValueError:
            print("Assistant: /topk expects a positive integer.")
            return top_k, recursive_default, False

        top_k = value
        print(f"Assistant: Using top_k={top_k} for subsequent answers.")
        return top_k, recursive_default, False

    if name == "greeting":
        remainder = command.partition(" ")[2].strip()
        if not remainder:
            print(f"Assistant: Current greeting is '{bot.greeting}'.")
            return top_k, recursive_default, False
        bot.greeting = remainder
        print("Assistant: Greeting updated.")
        return top_k, recursive_default, False

    print("Assistant: Unknown command. Type /help for assistance.")
    return top_k, recursive_default, False


def _chat_loop(bot: RAGChatbot, *, top_k: int, recursive_default: bool) -> int:
    current_top_k = max(1, top_k)
    current_recursive = recursive_default

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            current_top_k, current_recursive, should_exit = _handle_command(
                bot, user_input[1:], current_top_k, current_recursive
            )
            if should_exit:
                print("Assistant: See you later!")
                return 0
            continue

        reply = bot.answer(user_input, top_k=current_top_k)
        if not reply:
            reply = "I could not understand that. Could you rephrase?"
        print(f"Assistant: {reply}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code

    bot = RAGChatbot(
        default_source=args.default_source,
        greeting=args.greeting,
        top_k=args.top_k,
    )

    recursive_default = not args.no_recursive
    chat_kwargs = {
        "reset": args.reset,
        "recursive": recursive_default,
    }
    if args.source:
        chat_kwargs["source"] = args.source

    bot.create_chat(**chat_kwargs)
    _print_startup(bot)

    return _chat_loop(bot, top_k=args.top_k, recursive_default=recursive_default)


if __name__ == "__main__":
    sys.exit(main())
