# Phase 2 — Decision/Orchestrator Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Decision/Orchestrator Agent (Claude via API) that consumes the quant signal plus a stubbed research context and produces the trade recommendation with a fully journaled reasoning trail — still NVDA-only.

**Architecture:** New `tenx/agents/` package: `research_stub.py` (explicit placeholder context — Phase 4 replaces it) and `decision.py` (the LLM agent: prompt builder as a pure function, `decide()` with an injectable client, structured JSON output, full reasoning trail captured on the returned `Decision`). The pipeline chain becomes data → quant signal → research stub → decision → paper trade, journaling five stages; the *decision's* action (not the raw signal's) now drives the trade. `paper.py` is refactored to take an explicit side/price so the decision layer is the authority. The deterministic risk layer between decision and execution is Phase 3 — for now, as in Phases 0–1, trades remain hypothetical journal entries.

**Tech Stack:** Existing stack plus `anthropic==1.2.0` (verified installing, 2026-08-28). Model: `claude-opus-5` (skill-mandated default), adaptive thinking (default on Opus 5), structured output via `output_config.format` raw JSON schema, server-side refusal fallbacks (`fallbacks: "default"` + beta `server-side-fallback-2026-07-01`) enabled by default per current API guidance.

**Spec:** `docs/BUILD_PLAN.md` — Phase 2 row of §6, agent rules §4.4, logging §4.6, guardrails §9.

## Global Constraints

- Exactly this one new LLM agent (spec §4.4: two agents total; the Research Agent is Phase 4 and only a stub exists here).
- The agent RECOMMENDS; it cannot size positions or bypass anything. Its output is a constrained enum + conviction + rationale. The deterministic risk layer (Phase 3) will sit downstream (spec §4.5, §9).
- No LLM touches numerical pattern recognition — the agent receives the quant signal's already-computed output (spec §9).
- Full reasoning trail journaled per call: system prompt, user message, raw response text, response id, usage, stop reason — not just the parsed decision (spec §4.4, §4.6).
- The prompt must disclose the quant model's documented weakness (Phase 1 validation: no edge over base rate) — the agent must not be led to over-trust the signal.
- Offline test suite: `decide()` takes an injectable client; live API tests are marked `llm` and deselected by default (no credentials exist on this machine — `ANTHROPIC_API_KEY` unset, no `ant` profile).
- Model id exactly `claude-opus-5`; no temperature/top_p (removed on Opus 5); check `stop_reason == "refusal"` before reading content.

---

### Task 1: Research context stub

**Files:**
- Create: `tenx/agents/__init__.py`, `tenx/agents/research_stub.py`
- Modify: `pyproject.toml` (add `anthropic==1.2.0` dep, `tenx.agents` package, `llm` marker, deselect `llm` by default)
- Test: `tests/test_research_stub.py`

**Interfaces:**
- Produces: `stub_research_context(ticker: str) -> dict` returning `{"ticker": ticker, "status": "stub", "summary": <fixed sentence saying no research is available until Phase 4>, "sources": []}`.

- [ ] **Step 1: pyproject changes + install**

Add `"anthropic==1.2.0"` to dependencies; add `"tenx.agents"` to packages; change pytest `addopts` to `-m 'not network and not llm'` and add marker `llm: tests that call the live Claude API (deselected by default; needs credentials)`. Then `uv pip install --python .venv/bin/python -e .` and refreeze `requirements.lock`.

- [ ] **Step 2: Failing test**

```python
# tests/test_research_stub.py
from tenx.agents.research_stub import stub_research_context


def test_stub_context_shape():
    ctx = stub_research_context("NVDA")
    assert ctx["ticker"] == "NVDA"
    assert ctx["status"] == "stub"
    assert ctx["sources"] == []
    assert "Phase 4" in ctx["summary"]
```

- [ ] **Step 3: Verify fail, implement, verify pass**

```python
# tenx/agents/research_stub.py
"""Placeholder research context. The real Research Agent (news/filings/
sentiment synthesis) is Phase 4 — spec docs/BUILD_PLAN.md section 6.
The stub is explicit about being a stub so the Decision Agent's prompt
never implies research was actually performed."""


def stub_research_context(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "status": "stub",
        "summary": (
            "No research context available: the Research Agent is not built "
            "until Phase 4. Treat qualitative factors as unknown."
        ),
        "sources": [],
    }
```

(Note: the test requires the literal string "Phase 4" in the summary.)

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: explicit research-context stub (Phase 4 placeholder)"`

---

### Task 2: Decision Agent

**Files:**
- Create: `tenx/agents/decision.py`
- Test: `tests/test_decision.py` (offline, fake client), plus one `llm`-marked live smoke test

**Interfaces:**
- Consumes: `Signal` (Phase 0/1), research context dict (Task 1).
- Produces:
  - `Decision` frozen dataclass: `ticker: str`, `action: str` (BUY/SELL/HOLD), `conviction: float` (0–1), `rationale: str`, `model: str`, `trail: dict` (system, user, response_text, response_id, stop_reason, usage: {input_tokens, output_tokens}), `to_dict()`.
  - `build_decision_prompt(signal: Signal, research: dict) -> tuple[str, str]` — (system, user) pure function.
  - `decide(signal: Signal, research: dict, client=None, model: str = MODEL_ID) -> Decision` — `client=None` constructs `anthropic.Anthropic()`; raises `RuntimeError` on refusal stop reason; raises `ValueError` on malformed/invalid JSON action or conviction out of range.
  - `MODEL_ID = "claude-opus-5"`, `DECISION_SCHEMA` (raw JSON schema, additionalProperties false, required [action, conviction, rationale]).

- [ ] **Step 1: Failing tests**

```python
# tests/test_decision.py
import json
from types import SimpleNamespace

import pytest

from tenx.agents.decision import DECISION_SCHEMA, Decision, build_decision_prompt, decide
from tenx.agents.research_stub import stub_research_context
from tenx.signals.base import Signal


def make_signal(action="BUY", p_up=0.67):
    return Signal(
        ticker="NVDA", as_of="2026-08-27", action=action, confidence=p_up,
        features={"model": "logistic", "p_up": p_up, "last_close": 227.98,
                  "horizon": 5},
        rationale="test signal rationale",
    )


class FakeClient:
    """Mimics the slice of anthropic.Anthropic that decide() touches."""

    def __init__(self, payload: dict, stop_reason: str = "end_turn"):
        text_block = SimpleNamespace(type="text", text=json.dumps(payload))
        self.response = SimpleNamespace(
            content=[text_block],
            stop_reason=stop_reason,
            stop_details=SimpleNamespace(category="test", explanation="x"),
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            _request_id="req_test123",
        )
        self.last_kwargs = None
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.response


def test_prompt_contains_signal_research_and_caveat():
    system, user = build_decision_prompt(make_signal(), stub_research_context("NVDA"))
    assert "BUY" in user and "0.67" in user  # quant signal action and p_up
    assert "stub" in user or "No research" in user  # research status visible
    assert "base rate" in system  # honest quant-weakness disclosure
    assert "risk" in system.lower()  # downstream deterministic risk layer named


def test_decide_parses_valid_response():
    fake = FakeClient({"action": "BUY", "conviction": 0.6, "rationale": "because"})
    decision = decide(make_signal(), stub_research_context("NVDA"), client=fake)
    assert isinstance(decision, Decision)
    assert decision.action == "BUY"
    assert decision.conviction == 0.6
    assert decision.rationale == "because"
    assert decision.model == "claude-opus-5"
    # full reasoning trail captured for the journal
    assert decision.trail["response_text"]
    assert decision.trail["system"] and decision.trail["user"]
    assert decision.trail["usage"] == {"input_tokens": 100, "output_tokens": 50}
    assert decision.trail["stop_reason"] == "end_turn"
    # request was built correctly
    assert fake.last_kwargs["model"] == "claude-opus-5"
    assert fake.last_kwargs["output_config"]["format"]["schema"] == DECISION_SCHEMA["schema"] or \
        fake.last_kwargs["output_config"]["format"] == DECISION_SCHEMA


def test_decide_rejects_bad_action():
    fake = FakeClient({"action": "YOLO", "conviction": 0.6, "rationale": "x"})
    with pytest.raises(ValueError):
        decide(make_signal(), stub_research_context("NVDA"), client=fake)


def test_decide_rejects_out_of_range_conviction():
    fake = FakeClient({"action": "BUY", "conviction": 1.7, "rationale": "x"})
    with pytest.raises(ValueError):
        decide(make_signal(), stub_research_context("NVDA"), client=fake)


def test_decide_raises_on_refusal():
    fake = FakeClient({"action": "BUY", "conviction": 0.5, "rationale": "x"},
                      stop_reason="refusal")
    with pytest.raises(RuntimeError):
        decide(make_signal(), stub_research_context("NVDA"), client=fake)


def test_hold_is_a_valid_recommendation():
    fake = FakeClient({"action": "HOLD", "conviction": 0.8,
                       "rationale": "no research available, weak signal"})
    decision = decide(make_signal("HOLD", 0.5), stub_research_context("NVDA"), client=fake)
    assert decision.action == "HOLD"


@pytest.mark.llm
def test_decide_live_smoke():
    """Requires real credentials (ANTHROPIC_API_KEY or ant profile)."""
    decision = decide(make_signal(), stub_research_context("NVDA"))
    assert decision.action in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= decision.conviction <= 1.0
    assert len(decision.rationale) > 20
    assert decision.trail["response_id"]
```

- [ ] **Step 2: Verify fail (ModuleNotFoundError), implement**

```python
# tenx/agents/decision.py
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
```

- [ ] **Step 3: Run offline tests, verify 6 pass + 1 deselected (`llm`)**
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: Decision Agent (claude-opus-5, structured output, full reasoning trail)"`

---

### Task 3: Paper-trade refactor + pipeline integration

**Files:**
- Modify: `tenx/paper.py` (decision-driven signature), `tenx/pipeline.py` (five stages), `tests/test_paper.py`, `tests/test_pipeline.py`

**Interfaces:**
- `build_paper_trade(side: str, ticker: str, price: float, as_of: str, notional_usd: float = 10_000.0) -> dict | None` — returns `None` when `side == "HOLD"`; same dict shape as before (`signal_as_of` key keeps its name for journal continuity).
- `run_pipeline(..., research_fn=stub_research_context, decide_fn=decide)` — stages journaled in order: `data_pull`, `signal`, `research_context`, `decision` (payload = `decision.to_dict()`, trail included), `paper_trade`. The trade's side comes from **decision.action**; price/as_of from the signal's features. Returns `{"run_id", "ticker", "signal", "research", "decision", "trade"}`.

- [ ] **Step 1: Update `tests/test_paper.py`** — same four cases, called with explicit args (`build_paper_trade("BUY", "NVDA", 200.0, "2026-08-27")` etc.); HOLD → None; price > notional → ValueError.

- [ ] **Step 2: Update `tests/test_pipeline.py`** — inject `decide_fn` fakes:

```python
def fake_decide_buy(signal, research):
    from tenx.agents.decision import Decision
    return Decision(ticker=signal.ticker, action="BUY", conviction=0.7,
                    rationale="fake decision", model="fake",
                    trail={"system": "s", "user": "u", "response_text": "{}",
                           "response_id": "req_fake", "stop_reason": "end_turn",
                           "usage": {}})
```

End-to-end test asserts stages `["data_pull", "signal", "research_context", "decision", "paper_trade"]`, one run_id, decision journal payload contains `trail` with non-empty `system`, and the trade side equals the fake decision's action even if it differs from the signal's action (proving the decision is authoritative). HOLD-decision test asserts `trade is None` + journaled reason.

- [ ] **Step 3: Verify fails, implement `paper.py` + `pipeline.py` changes, verify all pass**

`tenx/paper.py` becomes:

```python
"""Hypothetical paper trades for the pre-Phase-3 pipeline. The side comes
from the Decision Agent's recommendation; no broker is involved — real IBKR
paper execution and the deterministic risk layer arrive in Phase 3."""


def build_paper_trade(
    side: str, ticker: str, price: float, as_of: str,
    notional_usd: float = 10_000.0,
) -> dict | None:
    if side == "HOLD":
        return None
    price = float(price)
    quantity = int(notional_usd // price)
    if quantity < 1:
        raise ValueError(
            f"price {price} exceeds notional {notional_usd}; no whole share affordable"
        )
    return {
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "price": price,
        "notional_usd": notional_usd,
        "signal_as_of": as_of,
        "hypothetical": True,
    }
```

`pipeline.py` — after the signal stage:

```python
    research = research_fn(ticker)
    journal.log(run_id, "research_context", research)

    decision = decide_fn(signal, research)
    journal.log(run_id, "decision", decision.to_dict())

    trade = build_paper_trade(
        decision.action, ticker,
        price=signal.features["last_close"], as_of=signal.as_of,
    )
    if trade is None:
        journal.log(run_id, "paper_trade", {
            "trade": None,
            "reason": f"decision was {decision.action}; no trade taken",
        })
    else:
        journal.log(run_id, "paper_trade", {"trade": trade})

    return {"run_id": run_id, "ticker": ticker, "signal": signal.to_dict(),
            "research": research, "decision": decision.to_dict(), "trade": trade}
```

with imports `from tenx.agents.decision import decide` / `from tenx.agents.research_stub import stub_research_context` and parameters `research_fn: Callable = stub_research_context, decide_fn: Callable = decide`.

- [ ] **Step 4: Full suite green; live-pipeline NOTE**

`.venv/bin/python -m pytest -q` → all pass, `network`+`llm` deselected. A live `python -m tenx.pipeline NVDA` now requires Anthropic credentials — cannot run on this machine today; document in the final report instead of faking it.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: decision-driven pipeline with five journaled stages; Phase 2 complete"`

---

## Self-Review Notes

- Spec §4.4 coverage: one Decision Agent on Claude via API, consuming quant signal + (stubbed) research, full per-call reasoning trail journaled. §4.6: five stages each log full payloads. §9: agent recommends only; deterministic sizing stays in `paper.py`; risk caps are explicitly deferred to Phase 3 (per the build order), and the prompt tells the agent it cannot waive them.
- The `llm` marker keeps the suite offline-green; the live smoke test is the user's one-command verification once credentials exist.
- Type consistency: `Decision.to_dict()` output is what both the journal payload and `run_pipeline`'s return use; `build_paper_trade`'s new signature is used identically in pipeline and tests.
