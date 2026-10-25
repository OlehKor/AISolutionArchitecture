import argparse
import logging
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv

from parking_assistant.approval import AdminDeliveryError
from parking_assistant.assistant import ParkingAssistant, conversation_diagram
from parking_assistant.composition import (
    build_admin_channel,
    build_admin_service,
    build_assistant,
    build_database,
    build_guardrails,
    build_knowledge,
)
from parking_assistant.config import PROJECT_ROOT, Settings, get_settings
from parking_assistant.conversation import LlmConversationModel
from parking_assistant.database import ParkingDatabase
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
from parking_assistant.loadtest import (
    StubConversationModel,
    load_environment,
    render_load_report,
    run_administrator_load,
    run_chat_load,
    run_mcp_load,
)
from parking_assistant.mcp_server import MIN_TOKEN_LENGTH, serve
from parking_assistant.models import build_chat_model, build_embeddings
from parking_assistant.workflow import workflow_diagram

logger = logging.getLogger("parking_assistant")
EXIT_WORDS = {"exit", "quit"}
QUIET_COMMANDS = {"chat", "run", "loadtest"}
GREETING = (
    "Hello! I'm the SkyPark Central parking assistant. I can answer questions about the garage "
    "(opening hours, prices, location, services) and reserve a parking space for you. "
    "How can I help? (Type 'exit' to leave.)"
)
DECISION_CHECK_SECONDS = 3
APOLOGY = "Sorry, I'm having trouble answering right now. Please try again in a moment."


def announce_decisions(
    assistant: ParkingAssistant,
    session_id: str,
    open_database: Callable[[], ParkingDatabase],
    stop: threading.Event,
    interval: float = DECISION_CHECK_SECONDS,
) -> None:
    """Print administrator decisions as soon as they arrive, without waiting for the next message."""
    database = open_database()
    try:
        while not stop.wait(interval):
            try:
                news = assistant.announcements(session_id, database)
            except Exception:
                logger.exception("Could not check for administrator decisions")
                continue
            if news:
                print(f"\nAssistant: {news}\nYou: ", end="", flush=True)
    finally:
        database.close()


def reply_safely(assistant: ParkingAssistant, session_id: str, text: str) -> str:
    """A failing turn (LLM, network, database) must not end the customer's conversation."""
    try:
        return assistant.reply(session_id, text)
    except Exception as error:
        # Provider error messages can echo parts of credentials, so only the type is logged.
        logger.warning("Could not answer a message (%s)", type(error).__name__)
        logger.debug("Details of the failed turn", exc_info=True)
        return APOLOGY


def chat(settings: Settings) -> None:
    if not settings.telegram_configured:
        logger.warning("Telegram is not configured: reservation requests will stay pending.")
    assistant = build_assistant(settings)
    session_id = str(uuid4())
    stop = threading.Event()
    threading.Thread(
        target=announce_decisions,
        args=(assistant, session_id, lambda: build_database(settings), stop),
        daemon=True,
    ).start()
    print(f"Assistant: {GREETING}")
    try:
        while True:
            try:
                text = input("You: ").strip()
            except EOFError:
                return
            if text.lower() in EXIT_WORDS:
                return
            if text:
                print(f"Assistant: {reply_safely(assistant, session_id, text)}")
    finally:
        stop.set()


def discover_chat_id(settings: Settings) -> None:
    channel = build_admin_channel(settings)
    print("Send any message to your bot in Telegram now (Ctrl+C to cancel)...")
    try:
        while not (messages := channel.poll()):
            pass
    except AdminDeliveryError as error:
        sys.exit(f"Could not read from Telegram: {error}")
    print(f"Your chat id is {messages[0].chat_id}.")
    print("Set PARKING_TELEGRAM_ADMIN_CHAT_ID to it in .env, then run `parking-assistant admin`.")


def admin(settings: Settings) -> None:
    if not settings.telegram_token_set:
        sys.exit("Set PARKING_TELEGRAM_BOT_TOKEN in .env first.")
    if not settings.telegram_configured:
        discover_chat_id(settings)
        return
    service = build_admin_service(settings, build_database(settings))
    resent = service.desk.resend_unsent()
    logger.info("Administrator agent is listening (%d unsent requests delivered)", resent)
    if not settings.mcp_token.get_secret_value():
        logger.warning("PARKING_MCP_TOKEN is not set: confirmed reservations will not be archived.")
    try:
        service.agent.serve()
    except KeyboardInterrupt:
        logger.info("Administrator agent stopped")


def mcp_server(settings: Settings, quiet: bool = False) -> None:
    try:
        serve(
            settings.mcp_token.get_secret_value(),
            settings.reservations_file,
            settings.mcp_host,
            settings.mcp_port,
            settings.mcp_extra_hosts,
            quiet=quiet,
        )
    except ValueError as error:
        sys.exit(str(error))


def ingest(settings: Settings) -> None:
    count = ingest_static_data(settings, build_embeddings(settings))
    logger.info("Stored %d chunks in collection %s", count, settings.milvus_collection)


def run_all(settings: Settings) -> None:
    """One terminal for everything: MCP server and administrator agent behind the chat."""
    token = settings.mcp_token.get_secret_value()
    if len(token) >= MIN_TOKEN_LENGTH:
        threading.Thread(target=mcp_server, args=(settings, True), daemon=True, name="mcp").start()
        print(f"MCP server started on {settings.mcp_url}")
    else:
        print(
            "PARKING_MCP_TOKEN is missing or too short: approvals will not be written to the file."
        )
    if settings.telegram_configured:
        threading.Thread(target=admin, args=(settings,), daemon=True, name="admin").start()
        print("Administrator agent started (Telegram)")
    else:
        print("Telegram is not configured: reservation requests will stay pending.")
    chat(settings)


def show_graphs() -> None:
    print("%% Conversation graph (one run per chat message)")
    print(conversation_diagram())
    print("%% Reservation workflow graph (one durable run per reservation)")
    print(workflow_diagram())


def loadtest(settings: Settings, args: argparse.Namespace) -> None:
    model_factory = (
        (lambda: LlmConversationModel(build_chat_model(settings)))
        if args.real_llm
        else StubConversationModel
    )
    with load_environment(
        settings.schema_sql,
        settings.seed_sql,
        build_knowledge(settings),
        build_guardrails(settings),
        model_factory,
    ) as environment:
        results = [
            run_chat_load(environment, args.sessions, args.concurrency, natural=args.real_llm),
            run_administrator_load(environment, args.concurrency),
        ]
    results.append(run_mcp_load(args.mcp_requests, args.mcp_concurrency))
    parameters = (
        f"{args.sessions} chat sessions at concurrency {args.concurrency} "
        f"({'real LLM' if args.real_llm else 'stub LLM'}), "
        f"{args.mcp_requests} MCP calls at each of {args.mcp_concurrency} parallel clients"
    )
    report = render_load_report(results, parameters, datetime.now())
    path = settings.reports_dir / "load_test_report.md"
    write_report(path, report)
    print(report)
    logger.info("Report written to %s", path)


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
    commands.add_parser("mcp-server", help="run the MCP server that stores confirmed reservations")
    commands.add_parser("run", help="run chat, administrator agent and MCP server in one terminal")
    commands.add_parser("graph", help="print the Mermaid diagrams of both LangGraph graphs")
    load = commands.add_parser("loadtest", help="load-test the chatbot, approvals and MCP server")
    load.add_argument("--sessions", type=int, default=24, help="concurrent-capable chat sessions")
    load.add_argument(
        "--concurrency", type=int, default=8, help="worker threads for chat and approvals"
    )
    load.add_argument("--mcp-requests", type=int, default=100, help="MCP calls per client count")
    load.add_argument(
        "--mcp-concurrency", type=int, nargs="+", default=[1, 4, 16], help="parallel MCP clients"
    )
    load.add_argument("--real-llm", action="store_true", help="use the real LLM (costs API calls)")
    commands.add_parser("ingest", help="(re)build the vector store from data/static")
    evaluation = commands.add_parser("evaluate", help="measure retrieval, latency and guard rails")
    evaluation.add_argument(
        "--end-to-end", action="store_true", help="also time full LLM answers (needs an API key)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    level = logging.WARNING if args.command in QUIET_COMMANDS else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    # Milvus Lite does not implement AllocTimestamp; pymilvus falls back, but grpc logs it as ERROR.
    logging.getLogger("grpc._server").setLevel(logging.CRITICAL)
    # Telegram request URLs contain the bot token and httpx logs them at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_settings()
    if args.command == "chat":
        chat(settings)
    elif args.command == "admin":
        admin(settings)
    elif args.command == "mcp-server":
        mcp_server(settings)
    elif args.command == "run":
        run_all(settings)
    elif args.command == "graph":
        show_graphs()
    elif args.command == "loadtest":
        loadtest(settings, args)
    elif args.command == "ingest":
        ingest(settings)
    else:
        evaluate(settings, args.end_to_end)


if __name__ == "__main__":
    main()
