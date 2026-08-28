from tenx.portfolio import Portfolio


def test_new_portfolio_defaults(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    assert p.cash == 100_000.0
    assert p.position_qty("NVDA") == 0
    assert p.equity({}) == 100_000.0
    assert p.to_dict()["equity_peak"] == 100_000.0


def test_buy_and_sell_accounting(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    p.apply_fill("NVDA", "BUY", 40, 200.0)
    assert p.cash == 100_000.0 - 8_000.0
    assert p.position_qty("NVDA") == 40
    assert p.position_notional("NVDA", 210.0) == 40 * 210.0
    p.apply_fill("NVDA", "SELL", 15, 210.0)
    assert p.cash == 92_000.0 + 15 * 210.0
    assert p.position_qty("NVDA") == 25


def test_short_position_counts_in_gross_exposure(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    p.apply_fill("NVDA", "SELL", 10, 200.0)
    assert p.position_qty("NVDA") == -10
    assert p.gross_exposure({"NVDA": 200.0}) == 2_000.0
    # equity: short position subtracts market value, cash was credited
    assert p.equity({"NVDA": 200.0}) == 100_000.0


def test_equity_peak_only_ratchets_up(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    assert p.update_equity_peak(110_000.0) == 110_000.0
    assert p.update_equity_peak(90_000.0) == 110_000.0


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "pf.json"
    p = Portfolio(path)
    p.apply_fill("NVDA", "BUY", 40, 200.0)
    p.update_equity_peak(101_000.0)
    p.save()
    p2 = Portfolio(path)
    assert p2.cash == p.cash
    assert p2.position_qty("NVDA") == 40
    assert p2.to_dict()["equity_peak"] == 101_000.0


def test_gross_exposure_falls_back_to_avg_price(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    p.apply_fill("NVDA", "BUY", 10, 200.0)
    assert p.gross_exposure({}) == 2_000.0  # no live price -> avg_price
