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
    assert fake.last_kwargs["output_config"]["format"] == DECISION_SCHEMA


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
