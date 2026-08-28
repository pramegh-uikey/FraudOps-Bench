import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "frozen_manifest.json"

# Every file whose content can change an arm's output for a given case:
# the SOP text (embedded in every prompt), the model/arm config, the
# prompt-building + calibration logic in flows.py, the LangGraph agent
# definition, and the selective-prediction band/self-consistency logic.
# If any of these differ from what's in the manifest, the methodology that
# produced a frozen calibration/holdout result is no longer the methodology
# on disk -- run_baseline.py's --split holdout guard refuses to run in
# that case (see verify_manifest below).
METHODOLOGY_FILES = [
    PROJECT_ROOT / "prompts" / "sop_v0.md",
    PROJECT_ROOT / "configs" / "models.yaml",
    PROJECT_ROOT / "src" / "flows.py",
    PROJECT_ROOT / "src" / "agentic_graph.py",
    PROJECT_ROOT / "src" / "selective_prediction.py",
    # Added 2026-08-28 alongside the GPT-5.6 Terra integration: this is
    # where every backend's actual call/retry/parsing logic lives (including
    # the new call_openai()), and was previously NOT covered here -- a real
    # integrity gap, since a silent edit to this file between calibration and
    # holdout would have gone completely undetected by the guard below.
    PROJECT_ROOT / "src" / "llm_backends.py",
    # Added for linear_retrieval/agentic_retrieval: the k-NN feature set and
    # k=5 must be frozen before calibration, same discipline as everything
    # else here -- tuning either against calibration accuracy later would
    # reproduce the exact small-sample overfitting mistake Section 5.3
    # documents, one level up.
    PROJECT_ROOT / "src" / "retrieval.py",
]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_hashes() -> dict[str, str]:
    return {str(p.relative_to(PROJECT_ROOT)): _hash_file(p) for p in METHODOLOGY_FILES}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def freeze(label: str | None = None) -> dict:
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "label": label,
        "file_hashes": _current_hashes(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_manifest() -> tuple[bool, str]:
    """Returns (ok, message). ok=False whenever no manifest exists yet, or
    any methodology file's current hash doesn't match what was frozen --
    i.e. the methodology has been edited since the freeze, so a holdout run
    against it would not be evaluating the methodology that was actually
    frozen and locked in."""
    if not MANIFEST_PATH.exists():
        return False, f"No frozen manifest found at {MANIFEST_PATH}. Run freeze_methodology.py first."

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    current = _current_hashes()
    frozen = manifest.get("file_hashes", {})

    mismatched = [f for f in current if current[f] != frozen.get(f)]
    if mismatched:
        return False, (
            "Methodology files changed since the manifest was frozen "
            f"(frozen_at={manifest.get('frozen_at')}): {mismatched}. "
            "Re-run freeze_methodology.py if this change is intentional and "
            "you accept it applies going forward, not retroactively to any "
            "already-reported holdout numbers."
        )

    return True, f"Manifest OK (frozen_at={manifest.get('frozen_at')}, git_commit={manifest.get('git_commit')})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None, help="free-text note, e.g. 'pre-calibration-run'")
    parser.add_argument("--verify", action="store_true", help="check the existing manifest instead of freezing a new one")
    args = parser.parse_args()

    if args.verify:
        ok, message = verify_manifest()
        print(message)
        raise SystemExit(0 if ok else 1)

    manifest = freeze(label=args.label)
    print(f"Froze methodology manifest to {MANIFEST_PATH}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
