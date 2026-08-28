from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Signal:
    ticker: str
    as_of: str  # ISO date of the last bar the signal was computed from
    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    features: dict = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
