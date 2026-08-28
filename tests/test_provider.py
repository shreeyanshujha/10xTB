import pandas as pd
import pytest

from tenx.data.openbb_provider import get_daily_bars

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@pytest.mark.network
def test_get_daily_bars_nvda_live():
    df = get_daily_bars("NVDA", start_date="2026-07-01")
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) > 10
    assert df.index.is_monotonic_increasing
    assert isinstance(df.index, pd.DatetimeIndex)  # downstream calls .date() on bars
    assert (df["close"] > 0).all()


@pytest.mark.network
def test_get_daily_bars_bad_ticker_raises():
    with pytest.raises(ValueError):
        get_daily_bars("ZZZZNOTATICKER", start_date="2026-07-01")
