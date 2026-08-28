"""The only module in the codebase allowed to import from openbb.

Every other layer consumes the standardized DataFrame returned here
(spec: docs/BUILD_PLAN.md section 4.2).
"""
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def get_daily_bars(
    ticker: str, start_date: str, end_date: str | None = None
) -> pd.DataFrame:
    from openbb import obb  # deferred: openbb import is slow (~seconds)

    try:
        result = obb.equity.price.historical(
            symbol=ticker,
            provider="yfinance",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        raise ValueError(f"no daily bars returned for {ticker!r}: {exc}") from exc

    df = result.to_dataframe()
    if df.empty:
        raise ValueError(f"no daily bars returned for {ticker!r}")
    df = df[REQUIRED_COLUMNS].sort_index()
    return df
