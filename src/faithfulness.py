import json
import re
from pathlib import Path

from splits import arm_output_path, evidence_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Matches generate_evidence_packets.py's tool_evidence keys.
TOOL_NAME_TO_EVIDENCE_KEY = {
    "get_transaction_details": "transaction_details",
    "get_card_history": "card_history",
    "get_email_domain_profile": "email_domain_profile",
    "get_device_history": "device_history",
    "get_velocity_summary": "velocity_summary",
    "get_identity_match_summary": "identity_match_summary",
}

# Free-text fields that may contain evidence claims worth checking.
CLAIM_FIELDS = ["evidence_used", "risk_indicators", "protective_indicators"]

_NUMBER_RE = re.compile(r"(?<![\w.%])[-+]?\$?\d[\d,]*\.?\d*\s*[%x]?(?![\w])")

# Patterns that look numeric but are never an evidentiary claim in this
# domain -- stripped from text before number extraction so they don't get
# scored as unverified (they're recapitulating a string field verbatim, not
# claiming a fact this scorer's numeric-only ground truth can check).
_IDENTITY_FLAG_RE = re.compile(r"\bid_\d+(?:\s*[/,-]\s*\d+)*\b", re.IGNORECASE)
_BROWSER_OS_VERSION_RE = re.compile(
    r"\b(?:chrome|firefox|safari|edge|opera|ie|windows|ios|ipados|android|"
    r"mobile safari|rv|trident|gecko|webkit)[:/\s]+[\d.]+\b",
    re.IGNORECASE,
)
_RESOLUTION_RE = re.compile(r"\b\d{2,5}x\d{2,5}\b", re.IGNORECASE)


def _strip_non_evidentiary_patterns(text: str) -> str:
    text = _IDENTITY_FLAG_RE.sub(" ", text)
    text = _BROWSER_OS_VERSION_RE.sub(" ", text)
    text = _RESOLUTION_RE.sub(" ", text)
    return text
_DOTTED_RE = re.compile(
    r"\b([a-z_]+)\.([a-z_][a-z0-9_.]*)\s*=\s*"
    r"([-+]?\$?\d[\d,]*\.?\d*%?|true|false|null|\"[^\"]*\"|[A-Za-z][A-Za-z0-9_]*)"
)

# Numbers this large are essentially always TransactionID/TransactionDT
# values incidentally mentioned in prose, not evidentiary claims -- none of
# the fields we score (rates, counts, ratios, dollar amounts) are this big
# in this dataset.
_ID_LIKE_THRESHOLD = 100_000


def load_jsonl(path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _index_by_case_id(rows: list[dict]) -> dict[str, dict]:
    return {r["case_id"]: r for r in rows}


def load_linear_evidence_by_case(split: str) -> dict[str, dict[str, dict]]:
    """case_id -> {evidence_key: evidence_dict} for the linear_api arm,
    straight from the split's evidence packets (already keyed correctly)."""
    rows = load_jsonl(evidence_path(split))
    return {r["case_id"]: r["tool_evidence"] for r in rows}


def load_agentic_evidence_by_case(split: str, arm: str = "agentic_api") -> dict[str, tuple[dict[str, dict], set[str]]]:
    """case_id -> ({evidence_key: evidence_dict}, {tool_names actually called})
    for the agentic_api arm, parsed from each run's tool_trace. Only
    includes tools the agent actually invoked -- citing one it never called
    is a violation caught separately (n_uncalled_tool_citations)."""
    rows = load_jsonl(arm_output_path(split, arm))
    result = {}
    for r in rows:
        tool_trace = r.get("tool_trace") or []
        evidence = {}
        called = set()
        for entry in tool_trace:
            tool_name = entry.get("tool_name")
            if not tool_name:
                continue
            called.add(tool_name)
            try:
                output = json.loads(entry["output"])
            except Exception:
                continue
            key = TOOL_NAME_TO_EVIDENCE_KEY.get(tool_name, tool_name)
            evidence[key] = output
        result[r["case_id"]] = (evidence, called)
    return result


def flatten_numeric_values(evidence: dict) -> set[float]:
    """Every numeric leaf in the (possibly nested) evidence dict, plus a
    percentage-form variant for values that look like a fraction (0-1), so
    text phrasing like "45%" matches a stored 0.45 without over-generating
    variants for arbitrary large numbers (which would just create noise)."""
    values: set[float] = set()

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
        elif isinstance(obj, bool):
            return  # bool is an int subclass -- exclude explicitly
        elif isinstance(obj, (int, float)):
            v = float(obj)
            if v != v:  # NaN check (some tool outputs serialize pandas NaN as literal JSON NaN)
                return
            values.add(round(v, 6))
            values.add(round(v))
            if 0 <= v <= 1:
                values.add(round(v * 100, 4))
                values.add(round(v * 100))

    _walk(evidence)
    return values


def _numbers_match(claimed: float, truth_set: set[float]) -> bool:
    for gt in truth_set:
        tol = max(0.01, abs(gt) * 0.02)
        if abs(claimed - gt) <= tol:
            return True
    return False


def _clean_number_token(token: str) -> tuple[float, bool] | None:
    """Returns (value, was_percentage) or None if unparseable."""
    is_pct = "%" in token
    cleaned = token.replace("$", "").replace(",", "").replace("%", "").replace("x", "").replace("X", "").strip()
    if not cleaned or cleaned in ("-", "+", "."):
        return None
    try:
        return float(cleaned), is_pct
    except ValueError:
        return None


def extract_claimed_numbers(text: str) -> list[float]:
    """Numeric claims worth checking, normalized to the same space as
    flatten_numeric_values (percentage tokens stay in percentage form,
    since that form is also added to the ground-truth set for fractions).
    """
    claims = []
    text = _strip_non_evidentiary_patterns(text)
    for match in _NUMBER_RE.finditer(text):
        parsed = _clean_number_token(match.group())
        if parsed is None:
            continue
        value, _is_pct = parsed
        if abs(value) >= _ID_LIKE_THRESHOLD:
            continue
        claims.append(value)
    return claims


def extract_dotted_citations(evidence_used: list[str]) -> list[tuple[str, str, str]]:
    """Best-effort extraction of the strict `tool.field = value` citation
    style the SOP's worked examples encourage. Returns (tool_key, field_path,
    raw_value_str) triples. Not every citation follows this style -- that's
    expected and handled by the looser numeric check above."""
    citations = []
    for entry in evidence_used:
        for tool_key, field_path, value in _DOTTED_RE.findall(entry):
            citations.append((tool_key, field_path, value))
    return citations


def _resolve_field_path(evidence: dict, field_path: str):
    node = evidence
    for part in field_path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _values_equal(actual, claimed_str: str) -> bool:
    claimed_str = claimed_str.strip().strip('"')
    if isinstance(actual, bool):
        return claimed_str.lower() == str(actual).lower()
    if isinstance(actual, (int, float)):
        parsed = _clean_number_token(claimed_str)
        if parsed is None:
            return False
        value, is_pct = parsed
        target = value / 100 if is_pct and 0 <= float(actual) <= 1 else value
        tol = max(0.01, abs(float(actual)) * 0.02)
        return abs(float(actual) - target) <= tol
    if actual is None:
        return claimed_str.lower() == "null"
    return str(actual).strip().lower() == claimed_str.strip().lower()


def score_case(model_output: dict, ground_truth_evidence: dict[str, dict], tools_called: set[str] | None = None) -> dict:
    """Faithfulness score for one case. ground_truth_evidence is keyed by
    evidence name (e.g. "card_history"), same as TOOL_NAME_TO_EVIDENCE_KEY
    values. tools_called is only meaningful for agentic_api; pass None for
    linear_api (it always has all 6 by construction)."""
    all_text_parts = []
    for field in CLAIM_FIELDS:
        all_text_parts.extend(str(x) for x in (model_output.get(field) or []))
    for cv in model_output.get("check_verdicts") or []:
        if isinstance(cv, dict) and cv.get("detail"):
            all_text_parts.append(str(cv["detail"]))
    if model_output.get("final_case_note"):
        all_text_parts.append(str(model_output["final_case_note"]))

    truth_numbers = set()
    for ev in ground_truth_evidence.values():
        truth_numbers |= flatten_numeric_values(ev)

    n_verified = 0
    n_unverified = 0
    for text in all_text_parts:
        for claimed in extract_claimed_numbers(text):
            if _numbers_match(claimed, truth_numbers):
                n_verified += 1
            else:
                n_unverified += 1

    evidence_used = [str(x) for x in (model_output.get("evidence_used") or [])]
    dotted = extract_dotted_citations(evidence_used)
    n_dotted_correct = 0
    for tool_key, field_path, value in dotted:
        source = ground_truth_evidence.get(tool_key)
        if source is None:
            continue
        actual = _resolve_field_path(source, field_path)
        if actual is None:
            continue
        if _values_equal(actual, value):
            n_dotted_correct += 1

    known_keys = set(TOOL_NAME_TO_EVIDENCE_KEY.values()) | set(TOOL_NAME_TO_EVIDENCE_KEY.keys())
    n_uninformative = 0
    for entry in evidence_used:
        mentions_tool = any(k in entry for k in known_keys)
        has_number = len(extract_claimed_numbers(entry)) > 0 or len(_DOTTED_RE.findall(entry)) > 0
        if mentions_tool and not has_number:
            n_uninformative += 1

    n_uncalled_tool_citations = 0
    if tools_called is not None:
        for entry in evidence_used:
            for tool_name in TOOL_NAME_TO_EVIDENCE_KEY:
                short = TOOL_NAME_TO_EVIDENCE_KEY[tool_name]
                if (tool_name in entry or short in entry) and tool_name not in tools_called:
                    n_uncalled_tool_citations += 1

    n_claimed = n_verified + n_unverified
    return {
        "n_claimed_numbers": n_claimed,
        "n_verified": n_verified,
        "n_unverified": n_unverified,
        "verified_rate": (n_verified / n_claimed) if n_claimed else None,
        "n_dotted_citations": len(dotted),
        "n_dotted_correct": n_dotted_correct,
        "n_uninformative_citations": n_uninformative,
        "n_uncalled_tool_citations": n_uncalled_tool_citations,
    }
