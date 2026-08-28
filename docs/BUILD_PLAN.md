# Agentic Trading Platform — Build Plan

**Context for the agent building this:** This is a solo personal project (one developer), not a team effort and not for job-interview purposes. It's built to be genuinely comprehensive and well-tested — explicitly not another shallow GitHub trading-bot clone. Read this whole document before writing any code; the Non-Goals and Guardrails sections encode hard-won decisions from extensive upfront design discussion and should not be silently relitigated.

## 1. Vision

A multi-agent algorithmic trading system that fuses fundamentals and technicals, does research synthesis, and makes trade decisions with a full, inspectable reasoning trail. The edge this system is built for is **decision quality over a multi-hour-to-multi-day holding period** — not execution speed. Paper trading first, always; live capital only after a pre-committed validation bar is met.

## 2. Goals

- A working multi-agent architecture with genuinely separated concerns (research, decision-making, risk, execution)
- Rigorous validation: walk-forward model testing, multi-ticker paper trading across more than one market regime, a success bar decided *before* the test — not after
- Full reasoning/decision logging, so any trade (or non-trade) can be explained after the fact
- A system that generalizes across a basket, not one overfit to a single stock's personality

## 3. Non-Goals (explicit — do not silently reintroduce these)

- **Not competing on execution speed or HFT.** No colocation, no FPGA, no latency racing. Prop-firm-style speed requires physical proximity to exchange matching engines that a home setup cannot replicate at any engineering effort level — this was evaluated and explicitly ruled out.
- **Not "almost all stocks of a market" in v1.** Starting with a small, named basket (Section 5). Broader coverage is a later-phase concern, not a v1 requirement.
- **Not hosted on a Jetson Nano as the core compute.** A Jetson Nano cannot ingest and run multi-model inference across a broad basket, and this project already has better infrastructure available (Section 4.1). The Jetson may get an optional, minor future role as a single-model edge inference node — never the platform's brain.
- **Not using an LLM for numerical pattern recognition.** LLMs are for qualitative synthesis (news, filings, sentiment) and decision orchestration — not for finding statistical structure in price/volume time series. That's the quant/ML layer's job, using trained models, not model prompting.
- **Not making risk management or execution LLM agents.** These must be deterministic code with hard limits that cannot be reasoned around. An agent that can "decide" to override a risk cap defeats the purpose of having the cap.
- **Not moving to live capital based on one ticker, one quarter, or a single good result.** Sample size and regime coverage matter; see Section 8.

## 4. System Architecture

### 4.1 Infrastructure (already available or in progress — use this, don't reinvent it)

| Component | Role |
|---|---|
| **BlackICE** (homelab, headless Debian/Docker) | Primary always-on host for the whole system |
| **n8n** (already running on BlackICE) | Orchestration/scheduling layer — triggers data pulls, agent runs, logging on schedule |
| **RTX 5090 rig** (incoming) | Quant/ML model training + inference once available; BlackICE CPU is sufficient for initial narrow-basket work |
| **Interactive Brokers (via MCP/API)** | Paper trading execution now; live execution later, same interface |
| **Claude API / Claude Max** | Powers the two LLM agents |
| **Jetson Nano** | Deferred / optional. Possible future role: a single lightweight, already-validated model running always-on at the edge. Not part of v1, not core compute. |

### 4.2 Data Layer

- **OpenBB Open Data Platform**, self-hosted on BlackICE. Note: OpenBB wound down as a company in August 2026 and released its stack under a permissive open-source license — **fork and vendor a pinned version** rather than depending on any hosted service or expecting ongoing vendor support.
- Provider connectors: yfinance (free, prices), FMP or Tiingo (fundamentals — pick based on actual data quality once pulling real data, see Open Questions), a news/filings source for the Research Agent.
- This is the single standardized interface every other layer reads from — no component talks to a raw provider API directly.

### 4.3 Quant/ML Layer — trained models, NOT agents

- **Purpose:** numerical pattern recognition on price/volume/fundamentals, both short-term and longer-horizon patterns.
- **Method:** gradient-boosted trees (e.g. XGBoost) or similarly appropriate tabular/time-series models — not an LLM.
- **Validation:** walk-forward only. Train on window 1, test on the next window, roll forward, retrain, repeat. A single static train/test split is not acceptable — it's exactly the kind of overfit backtest that's already been identified as untrustworthy.
- **Output:** a structured, confidence-scored signal per ticker, consumed by the Decision Agent.

### 4.4 Agent Layer — exactly 2 LLM agents

- **Research Agent:** reads news, filings, and sentiment per ticker, synthesizes qualitative context. This is where LLM reasoning genuinely adds value.
- **Decision/Orchestrator Agent:** combines the quant model's signal with the Research Agent's context, produces a trade rationale and recommendation.
- Both run on Claude via API. Full reasoning trail logged for every call — not just the final output.

### 4.5 Deterministic Guardrail Layer — plain code, NOT agents, non-negotiable

- **Risk layer:** hard position-size caps, exposure limits, max-drawdown circuit breaker, kill switch. Sits between the Decision Agent's output and Execution. Cannot be bypassed or reasoned around by any agent — this is a hard rule, not a suggestion an LLM can override with a good argument.
- **Execution layer:** takes an approved trade, submits it via IBKR (paper first), logs the order. Plumbing, not a decision point.

### 4.6 Logging — cross-cutting, non-negotiable

Every layer logs its full reasoning, not just outcomes:
- Quant model: signal + confidence per ticker per run
- Research Agent: sources consulted + synthesis
- Decision Agent: full rationale for every recommendation (including "no trade")
- Risk layer: every check, pass or block, and why
- Execution: every order with a full causal chain back to the signal and rationale that produced it

Without this, a good or bad quarter is unexplainable — exactly the "nobody knows why it worked" failure mode this project is trying to avoid.

## 5. Test Basket

**NVDA, AMD, INTC** (semiconductors/AI), **LMT, RTX** (defense primes), **PLTR** (AI + defense crossover).

**Known limitation, stated explicitly so it isn't rediscovered the hard way:** this is two correlated clusters plus one crossover name, not six independent trials. A broad AI-sector move or a defense-news event will likely move several of these together. Good performance here validates "the system trades AI-hardware and defense-contractor names well," not "the system generalizes across the market" — know which claim any result actually supports.

**Required addition:** one genuinely uncorrelated, non-cyclical name — a consumer staple or utility. Candidates: **Procter & Gamble (PG)**, **Coca-Cola (KO)**, or **NextEra Energy (NEE)**. Pick one before Phase 5 (full-basket testing); see Open Questions.

## 6. Build Phases

Build in this order. Do not parallelize across phases or jump ahead — each phase should run end-to-end before the next is added. Single Claude Code terminal until Phase 5; parallel sessions only make sense once individual modules have defined interfaces to build against.

| Phase | Scope |
|---|---|
| **0 — Walking Skeleton** | One stock (start with NVDA — cleanest data availability). OpenBB data pull → placeholder/dumb signal (doesn't need to be smart) → log a hypothetical paper trade. Goal: prove the full pipe runs end-to-end before adding any sophistication. |
| **1 — Real Quant Layer** | Replace the placeholder signal with a real, walk-forward-validated model. Still one stock, still no agents. |
| **2 — Decision Agent** | Add the Decision/Orchestrator Agent consuming the quant signal. Research context can be stubbed for now. Still one stock. |
| **3 — Risk + Execution** | Add the deterministic risk layer (caps, kill switch) between decision and execution. Wire real IBKR paper-trading execution + logging. |
| **4 — Research Agent** | Add the Research Agent (news/filings/sentiment) feeding the Decision Agent's context. |
| **5 — Scale to Full Basket** | Extend the validated single-stock pipeline across the full basket (Section 5, including the uncorrelated addition). This is where parallel Claude Code sessions become appropriate — each module now has a real interface. |
| **6 — Paper Trading Validation Window** | Run the full basket paper-trading simultaneously (not serially) for a defined window — minimum one quarter, ideally spanning more than one volatility regime. Go/no-go bar (Section 7) must be locked in *before* this phase starts. |
| **7 — Live Capital** | Only if the Phase 6 bar is met. Start small, same risk caps enforced, full logging continues, re-evaluate on a rolling basis against the same metrics. |

## 7. Success Metrics / Go-No-Go Criteria

**These must be set numerically before Phase 6 begins — not chosen after seeing results, which would just be rationalizing whatever happened.** At minimum, define:
- A minimum Sharpe ratio (risk-adjusted return threshold)
- A win-rate floor
- A maximum acceptable drawdown
- Performance versus a simple buy-and-hold benchmark on the same basket over the same window

Claude Fable should propose concrete starting numbers for these (informed by the basket's typical volatility) for confirmation — but they need to be locked in and written down before Phase 6's paper-trading window starts.

## 8. Open Questions

- Which uncorrelated basket addition — PG, KO, or NEE? (Section 5)
- Specific quant model choice per prediction task — decide empirically during Phase 1, don't over-plan this now
- Exact numeric go/no-go thresholds for Phase 6 (Section 7)
- FMP vs. Tiingo for fundamentals — decide during Phase 0/1 based on actual data quality once pulling real data

## 9. Guardrails (non-negotiable — do not relitigate mid-build)

- Never let agent reasoning override the deterministic risk layer.
- Never skip walk-forward validation in favor of a single train/test split.
- Never treat one good backtest or one good paper-trading window as sufficient evidence for live capital.
- Never scale to the full basket, or start parallel work, before the single-stock walking skeleton runs end-to-end.
- Never use an LLM for the numerical pattern-recognition job — that belongs to the quant/ML layer only.
