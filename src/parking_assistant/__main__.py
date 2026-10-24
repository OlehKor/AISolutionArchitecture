import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv

from parking_assistant.composition import (
    build_administrator,
    build_assistant,
    build_database,
    build_guardrails,
    build_knowledge,
)
from parking_assistant.config import PROJECT_ROOT, Settings, get_settings
from parking_assistant.evaluation import (
    load_guardrail_dataset,
    load_retrieval_cases,
    render_report,
    run_end_to_end,
    run_guardrails,
    run_retrieval,
    write_report,
)
from parking_assistant.knowledge import ingest_static_data
from parking_assistant.models import build_embeddings

logger = logging.getLogger("parking_assistant")
EXIT_WORDS = {"exit", "quit"}


def chat(settings: Settings) -> None:
    if not settings.telegram_configured:
        logger.warning("Telegram is not configured: reservation requests will stay pending.")
    assistant = build_assistant(settings)
    session_id = str(uuid4())
    print("SkyPark Central assistant. Type 'exit' to leave.")
    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            return
        if text.lower() in EXIT_WORDS:
            return
        if text:
            print(f"Assistant: {assistant.reply(session_id, text)}")


def admin(settings: Settings) -> None:
    if not settings.telegram_configured:
        sys.exit("Set PARKING_TELEGRAM_BOT_TOKEN and PARKING_TELEGRAM_ADMIN_CHAT_ID in .env first.")
    agent = build_administrator(settings, build_database(settings))
    resent = agent.resend_unsent()
    logger.info("Administrator agent is listening (%d unsent requests delivered)", resent)
    try:
        agent.serve()
    except KeyboardInterrupt:
        logger.info("Administrator agent stopped")


def ingest(settings: Settings) -> None:
    count = ingest_static_data(settings, build_embeddings(settings))
    logger.info("Stored %d chunks in collection %s", count, settings.milvus_collection)


def evaluate(settings: Settings, with_llm: bool) -> None:
    knowledge = build_knowledge(settings)
    retrieval_cases = load_retrieval_cases(settings.eval_data_dir / "retrieval.json")
    retrieval = run_retrieval(retrieval_cases, knowledge)
    guardrails = run_guardrails(
        load_guardrail_dataset(settings.eval_data_dir / "guardrails.json"),
        build_guardrails(settings),
        knowledge,
    )
    end_to_end = (
        run_end_to_end(build_assistant(settings), [case.question for case in retrieval_cases])
        if with_llm
        else None
    )
    report = render_report(retrieval, guardrails, end_to_end, datetime.now())
    path = settings.reports_dir / "evaluation_report.md"
    write_report(path, report)
    print(report)
    logger.info("Report written to %s", path)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="parking-assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat", help="talk to the assistant in the terminal")
    commands.add_parser("admin", help="run the administrator agent (Telegram)")
    commands.add_parser("ingest", help="(re)build the vector store from data/static")
    evaluation = commands.add_parser("evaluate", help="measure retrieval, latency and guard rails")
    evaluation.add_argument(
        "--end-to-end", action="store_true", help="also time full LLM answers (needs an API key)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    # Milvus Lite does not implement AllocTimestamp; pymilvus falls back, but grpc logs it as ERROR.
    logging.getLogger("grpc._server").setLevel(logging.CRITICAL)
    # Telegram request URLs contain the bot token and httpx logs them at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_settings()
    if args.command == "chat":
        chat(settings)
    elif args.command == "admin":
        admin(settings)
    elif args.command == "ingest":
        ingest(settings)
    else:
        evaluate(settings, args.end_to_end)


if __name__ == "__main__":
    main()
