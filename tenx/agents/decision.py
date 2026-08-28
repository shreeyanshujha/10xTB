"""Decision/Orchestrator Agent (spec section 4.4) — one of exactly two LLM
agents. Combines the quant signal with research context and produces a trade
RECOMMENDATION. It does not size positions and nothing it says can bypass
the deterministic risk layer that sits between it and execution (Phase 3,
spec sections 4.5 and 9). Full reasoning trail is captured on Decision.trail
and journaled by the pipeline (spec section 4.6)."""
import json
from dataclasses import asdict, dataclass

from tenx.signals.base import Signal

MODEL_ID = "claude-opus-5"
VALID_ACTIONS = {"BUY", "SELL", "HOLD"}

DECISION_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "conviction": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["action", "conviction", "rationale"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """\
You are the decision agent of a systematic trading pipeline for a personal
paper-trading system. You receive (1) a quantitative model's signal and
(2) qualitative research context, and you produce one trade recommendation.

Honest context you must weigh:
- The quant model's walk-forward validation showed NO accuracy edge over the
  per-fold base rate on this ticker (mean edge -0.055 across 46 out-of-sample
  quarterly folds; AUC 0.523). Its probability is weak evidence, not truth.
- If the research context is a stub or empty, qualitative factors are UNKNOWN,
  not neutral-positive.

Your role and limits:
- Recommend exactly one of BUY, SELL, or HOLD with a conviction in [0, 1]
  and a rationale that cites the actual inputs you were given.
- HOLD (no trade) is a first-class recommendation and the correct one when
  evidence is weak or conflicting.
- You do not size positions and you cannot waive any limit: a deterministic
  risk layer with hard caps sits between your recommendation and execution
  and is not influenced by your output's wording.
- Never invent data you were not given."""


def build_decision_prompt(signal: Signal, research: dict) -> tuple[str, str]:
    user = (
        "Quant signal (structured output of the walk-forward-validated model):\n"
        f"{json.dumps(signal.to_dict(), indent=2)}\n\n"
        "Research context:\n"
        f"{json.dumps(research, indent=2)}\n\n"
        "Produce your trade recommendation for this ticker."
    )
    return SYSTEM_PROMPT, user


@dataclass(frozen=True)
class Decision:
    ticker: str
    action: str
    conviction: float
    rationale: str
    model: str
    trail: dict

    def to_dict(self) -> dict:
        return asdict(self)


def decide(
    signal: Signal,
    research: dict,
    client=None,
    model: str = MODEL_ID,
) -> Decision:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    system, user = build_decision_prompt(signal, research)
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": DECISION_SCHEMA},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"decision agent refused: "
            f"{getattr(details, 'category', None)} — "
            f"{getattr(details, 'explanation', None)}"
        )

    text = next(b.text for b in response.content if b.type == "text")
    payload = json.loads(text)

    action = payload.get("action")
    conviction = payload.get("conviction")
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action from decision agent: {action!r}")
    if not isinstance(conviction, (int, float)) or not 0.0 <= conviction <= 1.0:
        raise ValueError(f"conviction out of range: {conviction!r}")

    usage = getattr(response, "usage", None)
    return Decision(
        ticker=signal.ticker,
        action=action,
        conviction=float(conviction),
        rationale=str(payload.get("rationale", "")),
        model=model,
        trail={
            "system": system,
            "user": user,
            "response_text": text,
            "response_id": getattr(response, "_request_id", None),
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            },
        },
    )
