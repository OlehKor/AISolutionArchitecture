import re
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

INPUT_ENTITIES = (
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "SECRET",
)
OUTPUT_ENTITIES = INPUT_ENTITIES
SCORE_THRESHOLD = 0.4
REDACTION_MARK = "[REDACTED]"

PROMPT_INJECTION = "prompt_injection"
SENSITIVE_REQUEST = "sensitive_data_request"

_FLAGS = re.IGNORECASE | re.DOTALL
_ASKING = r"(?:what(?:'s| is| are)|give|tell|share|show|send|reveal|list|provide|leak)"
_STAFF = r"(?:owner|manager|director|staff|employees?|admin(?:istrator)?)(?:'s|s')?"

INJECTION_PATTERNS = tuple(
    re.compile(pattern, _FLAGS)
    for pattern in (
        r"\b(?:ignore|disregard|forget|override)\b.{0,40}\b(?:instructions?|rules?|prompts?|guidelines?)\b",
        r"\b(?:reveal|show|print|display|repeat|output)\b.{0,30}\b(?:system|hidden|initial|developer)\s+(?:prompt|message|instructions?)\b",
        r"\byou are now\b",
        r"\b(?:developer|debug|god|dan)\s+mode\b",
        r"\bjailbreak",
        r"\bpretend (?:to be|you are)\b",
        r"\bwithout (?:any )?(?:restrictions|filters|guardrails|rules)\b",
    )
)

SENSITIVE_REQUEST_PATTERNS = tuple(
    re.compile(pattern, _FLAGS)
    for pattern in (
        rf"\b{_ASKING}\b.{{0,60}}\b{_STAFF}\b.{{0,40}}\b(?:personal|private|home|mobile|cell|phone|e-?mail|address|password|credentials?)\b",
        r"\b(?:gate|door|panel|safe|admin|back-?office|portal|staff)\b.{0,25}\b(?:code|password|pin|credentials?|login)\b",
        r"\b(?:code|password|pin|credentials?|login)\b.{0,25}\b(?:gate|door|panel|safe|admin|back-?office|portal|staff)\b",
        r"\b(?:other|another|all|every|previous|vip|regular|key)\s+(?:customers?|clients?|drivers?|users?|guests?)\b.{0,40}\b(?:reservations?|bookings?|plates?|names?|data|details|list|phones?|e-?mails?|cards?)\b",
        r"\b(?:internal|confidential|private)\s+(?:documents?|notes|operations|files?|data)\b",
        r"\b(?:reservations?|bookings?|plates?|names?|phones?|e-?mails?)\b.{0,20}\b(?:of|for|from|by)\s+(?:other|another|all|every|previous|vip|regular)\s+(?:customers?|clients?|drivers?|users?|guests?)\b",
        r"\b(?:corporate card|card on file|cash collection)\b",
    )
)

REFUSALS = {
    PROMPT_INJECTION: "I can't follow instructions that try to change my rules. "
    "I'm happy to help with parking information and reservations.",
    SENSITIVE_REQUEST: "I can't share private or internal information such as staff contacts, "
    "access credentials or other customers' data. I can help with general parking "
    "information and reservations.",
}


@dataclass(frozen=True)
class Redaction:
    text: str
    entities: tuple[str, ...]


@dataclass(frozen=True)
class InputVerdict:
    allowed: bool
    text: str
    reason: str
    masked_entities: tuple[str, ...]


def find_violation(text: str) -> str:
    """Return the violated policy name, or an empty string when the text is acceptable."""
    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        return PROMPT_INJECTION
    if any(pattern.search(text) for pattern in SENSITIVE_REQUEST_PATTERNS):
        return SENSITIVE_REQUEST
    return ""


def _credential_recognizer() -> PatternRecognizer:
    pattern = Pattern(
        name="credential",
        regex=r"(?i)\b(?:password|passcode|pin|(?:gate|door|panel|safe|access) code)\s*(?:is|:|=)\s*\S+",
        score=0.85,
    )
    return PatternRecognizer(supported_entity="SECRET", patterns=[pattern])


class PiiRedactor:
    """Detects personal data with Presidio (spaCy NER + pattern recognizers) and masks it."""

    def __init__(self, allow_list: tuple[str, ...], spacy_model: str = "en_core_web_sm") -> None:
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        ).create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        self._analyzer.registry.add_recognizer(_credential_recognizer())
        self._anonymizer = AnonymizerEngine()
        self._allow_list = list(allow_list)

    def redact(self, text: str, entities: tuple[str, ...]) -> Redaction:
        findings = self._analyzer.analyze(
            text=text,
            language="en",
            entities=list(entities),
            allow_list=self._allow_list,
            score_threshold=SCORE_THRESHOLD,
        )
        if not findings:
            return Redaction(text=text, entities=())
        masked = self._anonymizer.anonymize(
            text=text,
            analyzer_results=findings,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": REDACTION_MARK})},
        )
        return Redaction(
            text=masked.text, entities=tuple(sorted({f.entity_type for f in findings}))
        )


class Guardrails:
    def __init__(self, redactor: PiiRedactor) -> None:
        self._redactor = redactor

    def check_input(self, text: str) -> InputVerdict:
        violation = find_violation(text)
        if violation:
            return InputVerdict(False, text, violation, ())
        redaction = self._redactor.redact(text, INPUT_ENTITIES)
        return InputVerdict(True, redaction.text, "", redaction.entities)

    def sanitize_output(self, text: str) -> Redaction:
        return self._redactor.redact(text, OUTPUT_ENTITIES)
