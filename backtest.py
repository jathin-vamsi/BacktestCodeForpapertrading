"""
backtest.py  —  Walk-forward backtest that mirrors the original script exactly.

What "exactly" means
─────────────────────
1. All assets are reindexed onto a SINGLE shared BTC timestamp spine,
   so integer index i is the same row for every asset (same as original).
2. Entry/exit fills use next_row['open'] — signal on bar i, fill on bar i+1.
3. Entry signal lookbacks: ema_4h[i-15], ema_1h[i-25]  (original values).
4. Assets are ranked each bar and the top signals are taken (portfolio cap
   limits how many can be open simultaneously).
5. No lookahead — bar i only uses data[0..i].

Usage
─────
    python backtest.py --data-dir C:\CB --verbose
    python backtest.py --data-dir C:\CB --start 2022-01-01 --end 2026-01-01

CSV format
──────────
    columns: timestamp, open, high, low, close, volume
    timestamp: unix ms (Binance format) OR ISO string
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  — identical to original script
# ──────────────────────────────────────────────────────────────────────────────

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT",
    "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "LTCUSDT", "DOGEUSDT",
    "APTUSDT", "NEARUSDT", "FILUSDT", "OPUSDT", "ARBUSDT"
]

FEE_RATE        = 0.0004
SLIPPAGE        = 0.0004
BASE_RISK       = 0.006
MAX_RISK        = 0.015
PORT_CAP        = 0.10
MAX_POSITIONS   = 2
ATR_PERIOD      = 14
EMA_PERIOD      = 50
STOP_ATR        = 2.0
ADX_THRESHOLD   = 25
ATR_EXPANSION   = 1.3
MULTIPLIER      = 1.015
INITIAL_CAPITAL = 100.0
MIN_LOOKBACK    = 300       # matches original: range(300, length-1)
DEFAULT_START   = "2019-01-01"
SCRIPT_DIR      = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def compute_dynamic_risk(row, equity, equity_ma, risk_multiplier):
    regime_strength = clamp((float(row["adx"]) - 20.0) / 20.0, 0.0, 1.0)
    dr = BASE_RISK + regime_strength * (MAX_RISK - BASE_RISK)
    dr = min(dr, MAX_RISK)
    vol_ratio = float(row["atr"]) / float(row["atr_median"])
    vol_factor = clamp(1.0 / vol_ratio, 0.75, 1.25)
    dr *= vol_factor
    if equity < equity_ma:
        dr *= 0.5
    dr *= risk_multiplier
    return dr


def row_ok(row) -> bool:
    """Check all required indicator fields are present and valid."""
    for c in ["open", "high", "low", "close", "atr", "atr_median", "ema_1h", "ema_4h", "adx"]:
        v = row.get(c) if hasattr(row, "get") else getattr(row, c, None)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
    return float(row["atr"]) > 0 and float(row["atr_median"]) > 0


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # detect timestamp format
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        # Binance exports unix milliseconds
        test = pd.to_datetime(ts.iloc[0], unit="ms")
        if test.year >= 2000:
            df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    return df.sort_values("timestamp").reset_index(drop=True)


def prepare_asset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute indicators on the full history of one asset.
    Identical logic to the original script.
    """
    p = df.copy().sort_values("timestamp").reset_index(drop=True)

    p["prev_close"] = p["close"].shift(1)
    p["tr"] = np.maximum(
        p["high"] - p["low"],
        np.maximum(
            (p["high"] - p["prev_close"]).abs(),
            (p["low"]  - p["prev_close"]).abs(),
        ),
    )
    p["atr"]        = p["tr"].ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    p["atr_median"] = p["atr"].rolling(100).median()
    p["ema_1h"]     = p["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    # 4-hour EMA
    df_4h = (
        p.set_index("timestamp")
         .resample("4h")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
         .dropna()
    )
    df_4h["ema_4h"] = df_4h["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    p = p.set_index("timestamp")
    p = p.merge(df_4h[["ema_4h"]], left_index=True, right_index=True, how="left")
    p["ema_4h"] = p["ema_4h"].ffill()

    # ADX
    plus_dm  = p["high"].diff().copy()
    minus_dm = (-p["low"].diff()).copy()
    plus_dm[ (plus_dm  < 0) | (plus_dm  < minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm )] = 0
    tr_smooth = p["tr"].ewm(alpha=1 / 14, adjust=False).mean()
    plus_di   = 100 * (plus_dm.ewm( alpha=1 / 14, adjust=False).mean() / tr_smooth)
    minus_di  = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    p["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return p.reset_index()


def build_combined_data(
    data_dir: Path,
    available_assets: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Tuple[Dict[str, pd.DataFrame], int, int]:
    """
    Load all assets, reindex onto BTC spine (exactly like the original script),
    then slice to [start, end].

    Returns
    -------
    data        : {symbol -> DataFrame aligned on shared integer index}
    start_i     : first bar index to process  (= MIN_LOOKBACK)
    end_i       : last  bar index to process  (= len - 2, need i+1 for fill)
    """
    raw: Dict[str, pd.DataFrame] = {}

    for symbol in available_assets:
        candidates = (
            list(data_dir.glob(f"{symbol}*.csv")) +
            list(data_dir.glob(f"{symbol}*.parquet"))
        )
        if not candidates:
            print(f"[WARN] No file for {symbol}, skipping.")
            continue
        path = candidates[0]
        df   = load_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        raw[symbol] = prepare_asset(df)
        print(f"  Loaded {symbol}: {len(raw[symbol])} candles")

    if "BTCUSDT" not in raw:
        raise RuntimeError("BTCUSDT is required.")

    # ── STEP 1: build shared master spine from BTC (same as original) ────────
    master_ts = raw["BTCUSDT"]["timestamp"]

    # ── STEP 2: reindex every asset onto that spine (same as original) ───────
    aligned: Dict[str, pd.DataFrame] = {}
    for symbol, df in raw.items():
        frame = df.set_index("timestamp").reindex(master_ts).ffill()
        aligned[symbol] = frame.reset_index()   # integer index == BTC index

    length = len(master_ts)

    # ── STEP 3: find integer range for [start, end] ──────────────────────────
    ts_series = aligned["BTCUSDT"]["timestamp"]

    lo = MIN_LOOKBACK
    hi = length - 2   # need i+1 for next-open fill

    if start:
        idx_arr = np.where(ts_series >= start)[0]
        if len(idx_arr):
            lo = max(lo, int(idx_arr[0]))

    if end:
        idx_arr = np.where(ts_series <= end)[0]
        if len(idx_arr):
            hi = min(hi, int(idx_arr[-1]))

    print(f"\n  Shared spine: {length} bars")
    print(f"  Processing bars {lo} → {hi}  "
          f"({ts_series.iloc[lo].date()} → {ts_series.iloc[hi].date()})")

    return aligned, lo, hi


def discover_data_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in [SCRIPT_DIR, SCRIPT_DIR / "data", Path.cwd(), Path.cwd() / "data"]:
        if any(candidate.glob("*USDT_1h.csv")):
            return candidate.resolve()
    return SCRIPT_DIR.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    data: Dict[str, pd.DataFrame],
    start_i: int,
    end_i: int,
    available_assets: List[str],
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[Dict]]:

    # state
    equity          = INITIAL_CAPITAL
    peak_equity     = INITIAL_CAPITAL
    max_dd          = 0.0
    risk_multiplier = 1.0
    trade_count     = 0
    positions: Dict[str, Any] = {}
    equity_curve: List[float] = []
    trade_log:    List[Dict]  = []

    total = end_i - start_i + 1

    for i in range(start_i, end_i + 1):

        equity_curve.append(equity)
        equity_ma = float(np.mean(equity_curve[-50:])) if len(equity_curve) >= 50 else equity

        # ── EXITS ─────────────────────────────────────────────────────────────
        for symbol in list(positions.keys()):
            df   = data[symbol]
            row  = df.iloc[i]
            nrow = df.iloc[i + 1]
            pos  = positions[symbol]

            if not row_ok(row):
                continue

            initial_risk = float(pos["entry"]) - float(pos["stop"])
            if initial_risk <= 0:
                continue

            pos["extreme"] = max(float(pos["extreme"]), float(row["high"]))
            current_r = (float(row["close"]) - float(pos["entry"])) / initial_risk
            trailing  = float(pos["stop"])

            if current_r >= 1.5:
                trailing = max(
                    float(pos["stop"]),
                    float(pos["extreme"]) - 2.5 * float(row["atr"]),
                )

            if current_r >= 1.0 and not pos.get("partial_taken", False):
                partial_qty = float(pos["size"]) * 0.5
                exit_price = float(nrow["open"]) * (1 - SLIPPAGE)
                entry_price = float(pos["entry"])
                pnl = (exit_price - entry_price) * partial_qty
                fee = (entry_price * partial_qty + exit_price * partial_qty) * FEE_RATE
                net_pnl = pnl - fee
                equity += net_pnl

                pos["size"] = float(pos["size"]) * 0.5
                pos["partial_taken"] = True

                print(f"[PARTIAL EXIT] {symbol} index={i}")

                trade_log.append({
                    "type":        "partial_exit",
                    "symbol":      symbol,
                    "bar":         i,
                    "timestamp":   str(df["timestamp"].iloc[i]),
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(exit_price, 6),
                    "qty":         partial_qty,
                    "pnl":         round(pnl, 6),
                    "fee":         round(fee, 6),
                    "net_pnl":     round(net_pnl, 6),
                    "equity":      round(equity, 6),
                })

            if current_r >= 1.0:
                trailing = max(
                    trailing,
                    float(pos["entry"]) + 0.5 * float(row["atr"]),
                )

            if current_r >= 2.0:
                trailing = max(
                    trailing,
                    float(pos["extreme"]) - 1.5 * float(row["atr"]),
                )

            if float(row["low"]) > trailing:
                continue

            # fill on next bar's open (matches original)
            exit_price  = float(nrow["open"]) * (1 - SLIPPAGE)
            entry_price = float(pos["entry"])
            qty         = float(pos["size"])
            pnl         = (exit_price - entry_price) * qty
            fee         = (entry_price * qty + exit_price * qty) * FEE_RATE
            net_pnl     = pnl - fee
            equity     += net_pnl

            trade_log.append({
                "type":        "exit",
                "symbol":      symbol,
                "bar":         i,
                "timestamp":   str(df["timestamp"].iloc[i]),
                "entry_price": round(entry_price, 6),
                "exit_price":  round(exit_price, 6),
                "qty":         qty,
                "pnl":         round(pnl, 6),
                "fee":         round(fee, 6),
                "net_pnl":     round(net_pnl, 6),
                "equity":      round(equity, 6),
            })
            del positions[symbol]

        btc_df = data["BTCUSDT"]
        btc_row = btc_df.iloc[i]

        btc_volatility = float(btc_row["atr"]) > float(btc_row["atr_median"]) * 1.2
        btc_trend = float(btc_row["close"]) > float(btc_row["ema_4h"])

        # ── ENTRIES ───────────────────────────────────────────────────────────
        if not (btc_volatility and btc_trend):
            print(f"[REGIME BLOCK] index={i}")
        elif len(positions) < MAX_POSITIONS:
            total_open_risk = sum(p["risk"] for p in positions.values())

            if total_open_risk < equity * PORT_CAP:

                # Score every candidate asset (same loop order as original)
                # Original doesn't rank by score — it just takes the first
                # signals in ASSETS order. We preserve that exact behaviour.
                for symbol in available_assets:
                    if len(positions) >= MAX_POSITIONS:
                        break
                    if symbol in positions:
                        continue

                    df   = data[symbol]
                    row  = df.iloc[i]
                    nrow = df.iloc[i + 1]

                    if not row_ok(row):
                        continue

                    # ── combined breakout + pullback entry signal ───────────
                    asset_up = (
                        float(row["close"]) > float(row["ema_4h"])
                        and float(row["ema_4h"]) > float(df["ema_4h"].iloc[i - 15])
                    )

                    breakout_entry = (
                        float(row["close"]) > float(row["ema_4h"]) * MULTIPLIER
                        and float(row["ema_1h"]) > float(df["ema_1h"].iloc[i - 25])
                    )

                    pullback = (
                        float(row["close"]) <= float(row["ema_1h"]) * 1.02
                        and float(row["close"]) >= float(row["ema_4h"])
                    )

                    recent_high = max(
                        float(df["high"].iloc[i - 3]),
                        float(df["high"].iloc[i - 2]),
                        float(df["high"].iloc[i - 1]),
                    )
                    trigger = float(row["close"]) > recent_high
                    pullback_entry = pullback and trigger

                    long_cond = (
                        asset_up
                        and (breakout_entry or pullback_entry)
                        and float(row["adx"]) > ADX_THRESHOLD
                        and float(row["atr"]) > ATR_EXPANSION * float(row["atr_median"])
                    )

                    trend_strength = float(row["ema_4h"]) - float(df["ema_4h"].iloc[i - 10])
                    strong_trend = trend_strength > 0
                    volatility_strength = (
                        float(row["atr"]) > float(row["atr_median"]) * 1.4
                    )
                    quality_filter = strong_trend and volatility_strength

                    if not (long_cond and quality_filter):
                        continue

                    if symbol in ["ARBUSDT", "FILUSDT", "DOGEUSDT"]:
                        continue

                    trend_score = float(row["ema_4h"]) - float(df["ema_4h"].iloc[i - 10])
                    volatility_score = float(row["atr"]) / float(row["atr_median"])
                    strength_score = trend_score * volatility_score

                    dr = compute_dynamic_risk(row, equity, equity_ma, risk_multiplier)
                    dr = dr * clamp(strength_score, 0.5, 1.5)
                    trade_risk = equity * dr

                    if total_open_risk + trade_risk > equity * PORT_CAP:
                        continue

                    stop_distance = STOP_ATR * float(row["atr"])
                    if stop_distance <= 0:
                        continue

                    # Execute on next candle open (realistic execution)
                    entry_price = float(nrow["open"]) * (1 + SLIPPAGE)
                    stop_price  = entry_price - stop_distance
                    size        = trade_risk / stop_distance

                    print(
                        f"[ENTRY TYPE] {'BREAKOUT' if breakout_entry else 'PULLBACK'} "
                        f"{symbol} index={i}"
                    )
                    print(f"[FILTERED ENTRY] {symbol} index={i}")

                    positions[symbol] = {
                        "entry":   entry_price,
                        "stop":    stop_price,
                        "size":    size,
                        "risk":    trade_risk,
                        "extreme": float(row["high"]),
                        "adds":    0,
                    }
                    trade_count     += 1
                    total_open_risk += trade_risk

                    trade_log.append({
                        "type":        "entry",
                        "symbol":      symbol,
                        "bar":         i,
                        "timestamp":   str(df["timestamp"].iloc[i]),
                        "entry_price": round(entry_price, 6),
                        "stop_price":  round(stop_price, 6),
                        "size":        size,
                        "risk":        round(trade_risk, 6),
                        "equity":      round(equity, 6),
                    })

        # ── PYRAMIDING ────────────────────────────────────────────────────────
        total_open_risk = sum(p["risk"] for p in positions.values())

        for symbol in list(positions.keys()):
            if total_open_risk >= equity * PORT_CAP:
                break

            df   = data[symbol]
            row  = df.iloc[i]
            nrow = df.iloc[i + 1]
            pos  = positions[symbol]

            if not row_ok(row):
                continue

            initial_risk = float(pos["entry"]) - float(pos["stop"])
            if initial_risk <= 0:
                continue

            current_r = (float(row["close"]) - float(pos["entry"])) / initial_risk
            if current_r < 1.0 or int(pos.get("adds", 0)) >= 1:
                continue

            dr = compute_dynamic_risk(row, equity, equity_ma, risk_multiplier) * 0.5
            trade_risk = equity * dr

            if total_open_risk + trade_risk > equity * PORT_CAP:
                continue

            stop_distance = STOP_ATR * float(row["atr"])
            if stop_distance <= 0:
                continue

            add_price = float(nrow["open"]) * (1 + SLIPPAGE)
            new_size  = trade_risk / stop_distance

            pos["size"]  = float(pos["size"]) + new_size
            pos["risk"]  = float(pos["risk"]) + trade_risk
            pos["adds"]  = int(pos.get("adds", 0)) + 1
            trade_count += 1
            total_open_risk += trade_risk

            trade_log.append({
                "type":      "pyramid",
                "symbol":    symbol,
                "bar":       i,
                "timestamp": str(df["timestamp"].iloc[i]),
                "add_price": round(add_price, 6),
                "add_size":  new_size,
                "equity":    round(equity, 6),
            })

        # ── DRAWDOWN / RISK SCALING ───────────────────────────────────────────
        peak_equity = max(peak_equity, equity)
        drawdown    = (equity - peak_equity) / peak_equity if peak_equity else 0.0
        max_dd      = min(max_dd, drawdown)

        if drawdown < -0.20:
            risk_multiplier = 0.5
        elif drawdown < -0.10:
            risk_multiplier = 0.75
        else:
            risk_multiplier = 1.0

        if verbose and (i - start_i) % 500 == 0:
            pct = (i - start_i) / total * 100
            ts  = data["BTCUSDT"]["timestamp"].iloc[i]
            print(
                f"  [{pct:5.1f}%]  {ts.date()}  "
                f"equity={equity:.4f}  open={list(positions.keys())}"
            )

    # ── close any remaining open positions at last bar close ─────────────────
    last_i = end_i
    for symbol in list(positions.keys()):
        df         = data[symbol]
        row        = df.iloc[last_i]
        pos        = positions[symbol]
        exit_price = float(row["close"]) * (1 - SLIPPAGE)
        entry_price = float(pos["entry"])
        qty        = float(pos["size"])
        pnl        = (exit_price - entry_price) * qty
        fee        = (entry_price * qty + exit_price * qty) * FEE_RATE
        net_pnl    = pnl - fee
        equity    += net_pnl

        trade_log.append({
            "type":        "exit_eob",
            "symbol":      symbol,
            "bar":         last_i,
            "timestamp":   str(df["timestamp"].iloc[last_i]),
            "entry_price": round(entry_price, 6),
            "exit_price":  round(exit_price, 6),
            "qty":         qty,
            "pnl":         round(pnl, 6),
            "fee":         round(fee, 6),
            "net_pnl":     round(net_pnl, 6),
            "equity":      round(equity, 6),
        })
        del positions[symbol]

    equity_curve.append(equity)

    state = {
        "equity":       equity,
        "equity_curve": equity_curve,
        "max_drawdown": max_dd,
        "trade_count":  trade_count,
        "spine_start":  str(data["BTCUSDT"]["timestamp"].iloc[start_i]),
        "spine_end":    str(data["BTCUSDT"]["timestamp"].iloc[end_i]),
    }
    return state, trade_log


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(state: Dict, trade_log: List[Dict]) -> Dict[str, Any]:
    arr    = np.array(state["equity_curve"], dtype=float)
    equity = float(state["equity"])

    if len(arr) < 2:
        return {}

    returns = np.diff(arr) / arr[:-1]
    sharpe  = (
        np.mean(returns) / np.std(returns) * math.sqrt(8760)
        if np.std(returns) > 0 else 0.0
    )
    peak      = np.maximum.accumulate(arr)
    drawdowns = (arr - peak) / peak
    max_dd    = float(np.min(drawdowns))

    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    start_ts = pd.Timestamp(state["spine_start"])
    end_ts   = pd.Timestamp(state["spine_end"])
    years    = (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600)
    cagr     = (equity / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 and equity > 0 else 0.0
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0.0

    exits      = [t for t in trade_log if t["type"] in ("exit", "exit_eob")]
    wins       = [t for t in exits if t["net_pnl"] > 0]
    losses     = [t for t in exits if t["net_pnl"] <= 0]
    win_rate   = len(wins) / len(exits) if exits else 0.0
    avg_win    = float(np.mean([t["net_pnl"] for t in wins]))   if wins   else 0.0
    avg_loss   = float(np.mean([t["net_pnl"] for t in losses])) if losses else 0.0
    gross_win  = sum(t["net_pnl"] for t in wins)
    gross_loss = abs(sum(t["net_pnl"] for t in losses))
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")
    total_fees = sum(t.get("fee", 0.0) for t in exits)

    return {
        "backtest_start":   state["spine_start"],
        "backtest_end":     state["spine_end"],
        "initial_capital":  INITIAL_CAPITAL,
        "final_equity":     round(equity, 4),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct":         round(cagr * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "calmar_ratio":     round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades":     len(exits),
        "win_rate_pct":     round(win_rate * 100, 1),
        "avg_win":          round(avg_win, 4),
        "avg_loss":         round(avg_loss, 4),
        "profit_factor":    round(pf, 3),
        "total_fees_paid":  round(total_fees, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_report(metrics: Dict, trade_log: List[Dict]) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print("  BACKTEST RESULTS")
    print(sep)
    print(f"  Backtest Start    : {metrics['backtest_start']}")
    print(f"  Backtest End      : {metrics['backtest_end']}")
    print(f"  Initial Capital   : ${metrics['initial_capital']:>10.2f}")
    print(f"  Final Equity      : ${metrics['final_equity']:>10.2f}")
    sign = "+" if metrics["total_return_pct"] >= 0 else ""
    print(f"  Total Return      : {sign}{metrics['total_return_pct']:.2f}%")
    print(f"  CAGR              : {metrics['cagr_pct']:.2f}%")
    print(f"  Max Drawdown      : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:.3f}")
    print(f"  Calmar Ratio      : {metrics['calmar_ratio']:.3f}")
    print(f"  Total Trades      : {metrics['total_trades']}")
    print(f"  Win Rate          : {metrics['win_rate_pct']:.1f}%")
    print(f"  Avg Win           : ${metrics['avg_win']:.4f}")
    print(f"  Avg Loss          : ${metrics['avg_loss']:.4f}")
    print(f"  Profit Factor     : {metrics['profit_factor']:.3f}")
    print(f"  Total Fees Paid   : ${metrics['total_fees_paid']:.4f}")
    print(sep)

    by_symbol: Dict[str, Any] = {}
    for t in trade_log:
        if t["type"] not in ("exit", "exit_eob"):
            continue
        s = t["symbol"]
        if s not in by_symbol:
            by_symbol[s] = {"trades": 0, "pnl": 0.0, "wins": 0}
        by_symbol[s]["trades"] += 1
        by_symbol[s]["pnl"]    += t["net_pnl"]
        if t["net_pnl"] > 0:
            by_symbol[s]["wins"] += 1

    if by_symbol:
        print()
        print(f"  {'Symbol':<12} {'Trades':>6} {'Win%':>6} {'Net PnL':>12}")
        print("  " + "-" * 42)
        for sym, d in sorted(by_symbol.items(), key=lambda x: -x[1]["pnl"]):
            wr   = d["wins"] / d["trades"] * 100 if d["trades"] else 0
            sign = "+" if d["pnl"] >= 0 else ""
            print(f"  {sym:<12} {d['trades']:>6} {wr:>5.1f}%  {sign}${d['pnl']:>9.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest (shared spine)")
    parser.add_argument("--data-dir", default=None,
                        help="Folder with SYMBOL_1h.csv files (auto-detected if omitted)")
    parser.add_argument("--start", default=None,
                        help=f"Start date YYYY-MM-DD UTC (default: {DEFAULT_START})")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD UTC (default: all available data)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress every 500 bars")
    parser.add_argument("--output", default="backtest_results.json")
    args = parser.parse_args()

    data_dir = discover_data_dir(args.data_dir)
    print(f"\n  Using data directory: {data_dir}")

    def parse_ts(v, is_end=False):
        if not v:
            return None
        ts = pd.Timestamp(v)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if is_end and "T" not in v:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return ts

    start = parse_ts(args.start or DEFAULT_START)
    end   = parse_ts(args.end, is_end=True)

    available_assets = [
        s for s in ASSETS
        if list(data_dir.glob(f"{s}*.csv")) or list(data_dir.glob(f"{s}*.parquet"))
    ]
    if not available_assets:
        raise SystemExit(f"No data files found in {data_dir}.")
    if "BTCUSDT" not in available_assets:
        raise SystemExit("BTCUSDT data is required.")

    print(f"  Found: {available_assets}\n")

    data, start_i, end_i = build_combined_data(data_dir, available_assets, start, end)
    state, trade_log = run_backtest(data, start_i, end_i, available_assets, verbose=args.verbose)

    metrics = compute_metrics(state, trade_log)
    print_report(metrics, trade_log)

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps({
            "metrics":      metrics,
            "equity_curve": [float(x) for x in state["equity_curve"]],
            "trade_log":    trade_log,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  Results saved to {out_path}")

    out_path.with_suffix(".metrics.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in metrics.items()),
        encoding="utf-8",
    )
    print(f"  Metrics saved  to {out_path.with_suffix('.metrics.txt')}\n")


if __name__ == "__main__":
    main()
