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
