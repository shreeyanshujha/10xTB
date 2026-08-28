from tenx.agents.research_stub import stub_research_context


def test_stub_context_shape():
    ctx = stub_research_context("NVDA")
    assert ctx["ticker"] == "NVDA"
    assert ctx["status"] == "stub"
    assert ctx["sources"] == []
    assert "Phase 4" in ctx["summary"]
