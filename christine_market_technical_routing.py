import warnings
from typing import TypedDict, Dict, Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

warnings.filterwarnings("ignore")


class State(TypedDict, total=False):
    query: str
    ticker: str
    task: str
    asset_type: str
    option_details: Dict[str, Any]
    user_profile: Dict[str, Any]
    market_data: Dict[str, Any]
    technical_signals: Dict[str, Any]
    options_data: Dict[str, Any]
    price_data: Any
    technical_full: Dict[str, Any]
    options_full: Dict[str, Any]
    news_sentiment: Dict[str, Any]
    llm_analysis: str
    risk_score: float
    weighted_signals: Dict[str, Any]
    reflection_count: int
    reflection_feedback: str
    data_retry_count: int
    combined_analysis: Dict[str, Any]
    data_quality: str
    option_context: Dict[str, Any]
    assignment_risk: str
    data_completeness: Dict[str, Any]
    decision: Dict[str, Any]


# ---------- helpers ----------

def _calculate_sortino(returns: pd.Series) -> float:
    """Sortino ratio — penalizes only downside volatility, so it handles fat-tailed return distributions better than Sharpe."""
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(252))


def _detect_volume_price_divergence(df: pd.DataFrame, lookback: int = 10) -> dict:
    """Flag bearish divergence (new high on fading volume) or bullish divergence (new low on fading volume)."""
    if len(df) < lookback + 5:
        return {"detected": False}

    recent = df.tail(lookback)
    prev = df.iloc[-(lookback * 2):-lookback]

    if prev.empty:
        return {"detected": False}

    recent_high = recent["Close"].max()
    prev_high = prev["Close"].max()
    recent_avg_vol = recent["Volume"].mean()
    prev_avg_vol = prev["Volume"].mean()

    recent_low = recent["Close"].min()
    prev_low = prev["Close"].min()

    if recent_high > prev_high and recent_avg_vol < prev_avg_vol * 0.85:
        return {
            "detected": True,
            "type": "bearish_divergence",
            "direction": "bearish",
            "detail": f"Price made new high (${recent_high:.2f} > ${prev_high:.2f}) "
                      f"but volume declined ({recent_avg_vol/1e6:.1f}M vs {prev_avg_vol/1e6:.1f}M) -> bearish divergence",
        }

    if recent_low < prev_low and recent_avg_vol < prev_avg_vol * 0.85:
        return {
            "detected": True,
            "type": "bullish_divergence",
            "direction": "bullish",
            "detail": f"Price made new low (${recent_low:.2f} < ${prev_low:.2f}) "
                      f"but volume declined ({recent_avg_vol/1e6:.1f}M vs {prev_avg_vol/1e6:.1f}M) -> selling exhaustion",
        }

    return {"detected": False}


def _merge_nearby_levels(levels: list, threshold_pct: float = 0.01) -> list:
    """Collapse support/resistance levels that sit within threshold_pct of each other, weighting by strength."""
    if not levels:
        return []

    sorted_levels = sorted(levels, key=lambda x: x["price"])
    merged = [sorted_levels[0].copy()]

    for level in sorted_levels[1:]:
        if abs(level["price"] - merged[-1]["price"]) / merged[-1]["price"] < threshold_pct:
            total_strength = merged[-1]["strength"] + level["strength"]
            merged[-1]["price"] = (
                merged[-1]["price"] * merged[-1]["strength"]
                + level["price"] * level["strength"]
            ) / total_strength
            merged[-1]["price"] = round(merged[-1]["price"], 2)
            merged[-1]["strength"] = total_strength
            merged[-1]["source"] = f"{merged[-1]['source']}+{level['source']}"
        else:
            merged.append(level.copy())

    return merged


# ---------- data fetchers ----------

def get_market_data(ticker: str, period: str = "3mo") -> dict:
    """Pull OHLCV from yfinance and derive return / volatility / drawdown / Sharpe / Sortino stats."""
    try:
        df = yf.download(ticker, period=period, progress=False)
    except Exception as e:
        return {"error": f"Failed to download data for {ticker}: {str(e)}"}

    if df.empty:
        return {"error": f"No data returned for {ticker}"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df["Daily_Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
    df["Rolling_Volatility"] = df["Daily_Return"].rolling(window=20).std() * np.sqrt(252)
    df["Rolling_Max"] = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] - df["Rolling_Max"]) / df["Rolling_Max"]

    returns_clean = df["Daily_Return"].dropna()
    current_price = float(df["Close"].iloc[-1])
    daily_vol = float(returns_clean.std())

    tail_returns = returns_clean[returns_clean <= returns_clean.quantile(0.05)]

    stats = {
        "current_price": current_price,
        "period_high": float(df["High"].max()),
        "period_low": float(df["Low"].min()),
        "total_return": float(df["Cumulative_Return"].iloc[-1]),
        "avg_daily_return": float(returns_clean.mean()),
        "daily_volatility": daily_vol,
        "annualized_volatility": float(daily_vol * np.sqrt(252)),
        "max_drawdown": float(df["Drawdown"].min()),
        "sharpe_ratio": float((returns_clean.mean() / daily_vol) * np.sqrt(252)) if daily_vol > 0 else 0.0,
        "sortino_ratio": _calculate_sortino(returns_clean),
        "var_95": float(returns_clean.quantile(0.05)),
        # Conditional VaR (expected loss beyond the 95% threshold).
        "cvar_95": float(tail_returns.mean()) if len(tail_returns) > 0 else 0.0,
        "skewness": float(returns_clean.skew()),
        "kurtosis": float(returns_clean.kurtosis()),
        "avg_daily_volume": float(df["Volume"].mean()),
        "recent_5d_return": float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) if len(df) >= 6 else 0.0,
        "recent_20d_return": float(df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) if len(df) >= 21 else 0.0,
        "trading_days": len(df),
        "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
    }
    return {"ticker": ticker, "price_data": df, "stats": stats}


def get_technical_indicators(df: pd.DataFrame) -> dict:
    """Compute SMA/EMA/RSI/MACD/Bollinger/Stoch/ADX/ATR/OBV/momentum, then turn them into bullish/bearish signal counts."""
    df = df.copy()

    df["SMA_10"] = ta.sma(df["Close"], length=10)
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)

    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd_df["MACD_12_26_9"]
    df["MACD_Signal"] = macd_df["MACDs_12_26_9"]
    df["MACD_Hist"] = macd_df["MACDh_12_26_9"]

    bbands = ta.bbands(df["Close"], length=20, std=2)
    for col in bbands.columns:
        if "BBL" in col:
            df["BB_Lower"] = bbands[col]
        elif "BBM" in col:
            df["BB_Middle"] = bbands[col]
        elif "BBU" in col:
            df["BB_Upper"] = bbands[col]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    df["Stoch_K"] = stoch["STOCHk_14_3_3"]
    df["Stoch_D"] = stoch["STOCHd_14_3_3"]

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX"] = adx_df["ADX_14"]
    df["DI_Plus"] = adx_df["DMP_14"]
    df["DI_Minus"] = adx_df["DMN_14"]

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["Momentum"] = ta.mom(df["Close"], length=10)
    df["ROC"] = ta.roc(df["Close"], length=10)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    current_price = float(latest["Close"])

    indicators = {
        "current_price": current_price,
        "sma_10": float(latest["SMA_10"]) if pd.notna(latest["SMA_10"]) else None,
        "sma_20": float(latest["SMA_20"]) if pd.notna(latest["SMA_20"]) else None,
        "sma_50": float(latest["SMA_50"]) if pd.notna(latest["SMA_50"]) else None,
        "ema_12": float(latest["EMA_12"]) if pd.notna(latest["EMA_12"]) else None,
        "ema_26": float(latest["EMA_26"]) if pd.notna(latest["EMA_26"]) else None,
        "rsi": float(latest["RSI"]) if pd.notna(latest["RSI"]) else None,
        "macd": float(latest["MACD"]) if pd.notna(latest["MACD"]) else None,
        "macd_signal": float(latest["MACD_Signal"]) if pd.notna(latest["MACD_Signal"]) else None,
        "macd_histogram": float(latest["MACD_Hist"]) if pd.notna(latest["MACD_Hist"]) else None,
        "stoch_k": float(latest["Stoch_K"]) if pd.notna(latest["Stoch_K"]) else None,
        "stoch_d": float(latest["Stoch_D"]) if pd.notna(latest["Stoch_D"]) else None,
        "bb_upper": float(latest["BB_Upper"]) if pd.notna(latest["BB_Upper"]) else None,
        "bb_middle": float(latest["BB_Middle"]) if pd.notna(latest["BB_Middle"]) else None,
        "bb_lower": float(latest["BB_Lower"]) if pd.notna(latest["BB_Lower"]) else None,
        "bb_width": float(latest["BB_Width"]) if pd.notna(latest["BB_Width"]) else None,
        "atr": float(latest["ATR"]) if pd.notna(latest["ATR"]) else None,
        "adx": float(latest["ADX"]) if pd.notna(latest["ADX"]) else None,
        "di_plus": float(latest["DI_Plus"]) if pd.notna(latest["DI_Plus"]) else None,
        "di_minus": float(latest["DI_Minus"]) if pd.notna(latest["DI_Minus"]) else None,
        "momentum": float(latest["Momentum"]) if pd.notna(latest["Momentum"]) else None,
        "roc": float(latest["ROC"]) if pd.notna(latest["ROC"]) else None,
    }

    signals = {}
    bullish_count = 0
    bearish_count = 0

    # Widen RSI bands when realized vol is high — fixed 30/70 generates too many false signals on volatile names.
    daily_returns = df["Close"].pct_change().dropna()
    annualized_vol = float(daily_returns.std() * np.sqrt(252)) if not daily_returns.empty else 0.0

    if annualized_vol < 0.30:
        rsi_oversold, rsi_overbought = 30, 70
        vol_regime = "low_vol"
    elif annualized_vol < 0.50:
        rsi_oversold, rsi_overbought = 25, 75
        vol_regime = "high_vol"
    else:
        rsi_oversold, rsi_overbought = 20, 80
        vol_regime = "extreme_vol"

    if indicators["rsi"] is not None:
        if indicators["rsi"] < rsi_oversold:
            signals["rsi"] = {
                "signal": "oversold",
                "direction": "bullish",
                "detail": (
                    f"RSI={indicators['rsi']:.1f}, below {rsi_oversold} "
                    f"({vol_regime}, ann. vol={annualized_vol:.0%}) "
                    f"-> oversold, potential bounce"
                ),
            }
            bullish_count += 1
        elif indicators["rsi"] > rsi_overbought:
            signals["rsi"] = {
                "signal": "overbought",
                "direction": "bearish",
                "detail": (
                    f"RSI={indicators['rsi']:.1f}, above {rsi_overbought} "
                    f"({vol_regime}, ann. vol={annualized_vol:.0%}) "
                    f"-> overbought, potential pullback"
                ),
            }
            bearish_count += 1
        else:
            signals["rsi"] = {
                "signal": "neutral",
                "direction": "neutral",
                "detail": (
                    f"RSI={indicators['rsi']:.1f}, in neutral zone "
                    f"({rsi_oversold}-{rsi_overbought}, {vol_regime})"
                ),
            }

    if indicators["macd"] is not None and indicators["macd_signal"] is not None:
        prev_macd_cross = float(prev["MACD"] - prev["MACD_Signal"]) if pd.notna(prev["MACD"]) else None

        if indicators["macd"] > indicators["macd_signal"]:
            direction = "bullish"
            bullish_count += 1
            if prev_macd_cross is not None and prev_macd_cross < 0:
                detail = "MACD just crossed ABOVE signal line -> fresh bullish crossover"
            else:
                detail = f"MACD({indicators['macd']:.2f}) > Signal({indicators['macd_signal']:.2f}) -> bullish"
        else:
            direction = "bearish"
            bearish_count += 1
            if prev_macd_cross is not None and prev_macd_cross > 0:
                detail = "MACD just crossed BELOW signal line -> fresh bearish crossover"
            else:
                detail = f"MACD({indicators['macd']:.2f}) < Signal({indicators['macd_signal']:.2f}) -> bearish"
        signals["macd"] = {
            "signal": "bullish_cross" if direction == "bullish" else "bearish_cross",
            "direction": direction,
            "detail": detail,
        }

    if all(indicators[k] is not None for k in ["sma_20", "sma_50"]):
        if current_price > indicators["sma_20"] > indicators["sma_50"]:
            signals["moving_averages"] = {
                "signal": "bullish_alignment", "direction": "bullish",
                "detail": f"Price(${current_price:.2f}) > SMA20(${indicators['sma_20']:.2f}) > SMA50(${indicators['sma_50']:.2f})",
            }
            bullish_count += 1
        elif current_price < indicators["sma_20"] < indicators["sma_50"]:
            signals["moving_averages"] = {
                "signal": "bearish_alignment", "direction": "bearish",
                "detail": f"Price(${current_price:.2f}) < SMA20(${indicators['sma_20']:.2f}) < SMA50(${indicators['sma_50']:.2f})",
            }
            bearish_count += 1
        else:
            signals["moving_averages"] = {
                "signal": "mixed", "direction": "neutral",
                "detail": f"Mixed: Price=${current_price:.2f}, SMA20=${indicators['sma_20']:.2f}, SMA50=${indicators['sma_50']:.2f}",
            }

    if indicators["bb_upper"] is not None and indicators["bb_lower"] is not None:
        if current_price > indicators["bb_upper"]:
            signals["bollinger"] = {
                "signal": "above_upper", "direction": "bearish",
                "detail": f"Price above upper band (${indicators['bb_upper']:.2f}) -> potential reversal down",
            }
            bearish_count += 1
        elif current_price < indicators["bb_lower"]:
            signals["bollinger"] = {
                "signal": "below_lower", "direction": "bullish",
                "detail": f"Price below lower band (${indicators['bb_lower']:.2f}) -> potential reversal up",
            }
            bullish_count += 1
        else:
            bb_position = (current_price - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"])
            signals["bollinger"] = {
                "signal": "within_bands", "direction": "neutral",
                "detail": f"Price within bands, position: {bb_position:.0%} (0%=lower, 100%=upper)",
            }

    if indicators["stoch_k"] is not None:
        if indicators["stoch_k"] > 80:
            signals["stochastic"] = {
                "signal": "overbought", "direction": "bearish",
                "detail": f"%K={indicators['stoch_k']:.1f}, above 80 -> overbought",
            }
            bearish_count += 1
        elif indicators["stoch_k"] < 20:
            signals["stochastic"] = {
                "signal": "oversold", "direction": "bullish",
                "detail": f"%K={indicators['stoch_k']:.1f}, below 20 -> oversold",
            }
            bullish_count += 1
        else:
            signals["stochastic"] = {
                "signal": "neutral", "direction": "neutral",
                "detail": f"%K={indicators['stoch_k']:.1f}, in neutral zone",
            }

    if indicators["adx"] is not None:
        trend_strength = "strong" if indicators["adx"] > 25 else "weak"
        trend_dir = "up" if (indicators.get("di_plus") or 0) > (indicators.get("di_minus") or 0) else "down"
        signals["adx"] = {
            "signal": f"{trend_strength}_trend_{trend_dir}",
            "direction": "bullish" if trend_dir == "up" and trend_strength == "strong"
                         else "bearish" if trend_dir == "down" and trend_strength == "strong"
                         else "neutral",
            "detail": (
                f"ADX={indicators['adx']:.1f} "
                f"({'strong trend' if trend_strength == 'strong' else 'weak/no trend'}), "
                f"+DI={indicators.get('di_plus', 0):.1f}, "
                f"-DI={indicators.get('di_minus', 0):.1f} -> trend direction: {trend_dir}"
            ),
        }
        if trend_strength == "strong":
            if trend_dir == "up":
                bullish_count += 1
            else:
                bearish_count += 1

    vp_div = _detect_volume_price_divergence(df)
    if vp_div["detected"]:
        signals["volume_divergence"] = {
            "signal": vp_div["type"],
            "direction": vp_div["direction"],
            "detail": vp_div["detail"],
        }
        if vp_div["direction"] == "bullish":
            bullish_count += 1
        elif vp_div["direction"] == "bearish":
            bearish_count += 1

    if "OBV" in df.columns and len(df) >= 20:
        obv_sma = df["OBV"].rolling(20).mean()
        if pd.notna(obv_sma.iloc[-1]):
            obv_current = float(df["OBV"].iloc[-1])
            obv_avg = float(obv_sma.iloc[-1])
            if obv_current > obv_avg * 1.05:
                signals["obv"] = {
                    "signal": "obv_rising", "direction": "bullish",
                    "detail": "OBV above 20-day average -> buying pressure increasing",
                }
                bullish_count += 1
            elif obv_current < obv_avg * 0.95:
                signals["obv"] = {
                    "signal": "obv_falling", "direction": "bearish",
                    "detail": "OBV below 20-day average -> selling pressure increasing",
                }
                bearish_count += 1
            else:
                signals["obv"] = {
                    "signal": "obv_flat", "direction": "neutral",
                    "detail": "OBV near 20-day average -> no clear volume trend",
                }

    signal_stats = {}
    if "RSI" in df.columns:
        signal_stats["rsi_overbought_days"] = int((df["RSI"] > rsi_overbought).sum())
        signal_stats["rsi_oversold_days"] = int((df["RSI"] < rsi_oversold).sum())

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd_diff = df["MACD"] - df["MACD_Signal"]
        signal_stats["macd_golden_crosses"] = int(((macd_diff > 0) & (macd_diff.shift(1) < 0)).sum())
        signal_stats["macd_death_crosses"] = int(((macd_diff < 0) & (macd_diff.shift(1) > 0)).sum())

    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        signal_stats["bb_above_upper_days"] = int((df["Close"] > df["BB_Upper"]).sum())
        signal_stats["bb_below_lower_days"] = int((df["Close"] < df["BB_Lower"]).sum())

    total = bullish_count + bearish_count
    bullish_pct = bullish_count / total * 100 if total > 0 else 50

    signal_summary = (
        f"Technical signals: {bullish_count} bullish, {bearish_count} bearish, "
        f"{len(signals) - bullish_count - bearish_count} neutral. "
        f"Overall lean: "
        f"{'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'NEUTRAL'} "
        f"({bullish_pct:.0f}% bullish). "
        f"RSI thresholds used: {rsi_oversold}/{rsi_overbought} ({vol_regime})."
    )

    return {
        "indicators": indicators,
        "signals": signals,
        "signal_stats": signal_stats,
        "signal_summary": signal_summary,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "rsi_thresholds": {
            "oversold": rsi_oversold,
            "overbought": rsi_overbought,
            "regime": vol_regime,
            "annualized_vol": round(annualized_vol, 4),
        },
        "full_df": df,
    }


def get_support_resistance(df: pd.DataFrame, options_result: dict = None,
                           window: int = 20) -> dict:
    """Build support/resistance levels by combining recent highs/lows, MAs, Bollinger, volume clusters, and (optional) options OI."""
    current_price = float(df["Close"].iloc[-1])
    supports = []
    resistances = []

    recent = df.tail(window)
    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())

    if recent_low < current_price:
        supports.append({"price": recent_low, "source": "recent_low", "strength": 2})
    if recent_high > current_price:
        resistances.append({"price": recent_high, "source": "recent_high", "strength": 2})

    for col, name in [("SMA_20", "SMA20"), ("SMA_50", "SMA50"), ("EMA_12", "EMA12")]:
        if col in df.columns and pd.notna(df[col].iloc[-1]):
            ma_val = float(df[col].iloc[-1])
            if ma_val < current_price:
                supports.append({"price": ma_val, "source": name, "strength": 2 if col == "SMA_50" else 1})
            elif ma_val > current_price:
                resistances.append({"price": ma_val, "source": name, "strength": 2 if col == "SMA_50" else 1})

    if "BB_Lower" in df.columns and pd.notna(df["BB_Lower"].iloc[-1]):
        bb_lower = float(df["BB_Lower"].iloc[-1])
        bb_upper = float(df["BB_Upper"].iloc[-1])
        supports.append({"price": bb_lower, "source": "BB_lower", "strength": 1})
        resistances.append({"price": bb_upper, "source": "BB_upper", "strength": 1})

    if len(recent) >= 10:
        price_bins = pd.cut(recent["Close"], bins=10)
        vol_by_price = recent.groupby(price_bins, observed=True)["Volume"].sum()
        if not vol_by_price.empty:
            top_vol_bin = vol_by_price.idxmax()
            vol_center = (top_vol_bin.left + top_vol_bin.right) / 2
            if vol_center < current_price:
                supports.append({"price": round(float(vol_center), 2),
                                 "source": "volume_cluster", "strength": 3})
            elif vol_center > current_price:
                resistances.append({"price": round(float(vol_center), 2),
                                    "source": "volume_cluster", "strength": 3})

    if options_result and "error" not in options_result:
        max_pain = options_result.get("max_pain")
        if max_pain:
            if max_pain < current_price:
                supports.append({"price": max_pain, "source": "max_pain", "strength": 3})
            elif max_pain > current_price:
                resistances.append({"price": max_pain, "source": "max_pain", "strength": 3})

        put_strike = options_result.get("max_put_oi_strike")
        if put_strike and put_strike < current_price:
            supports.append({"price": put_strike, "source": "put_wall", "strength": 3})

        call_strike = options_result.get("max_call_oi_strike")
        if call_strike and call_strike > current_price:
            resistances.append({"price": call_strike, "source": "call_wall", "strength": 3})

    supports = _merge_nearby_levels(supports, threshold_pct=0.01)
    resistances = _merge_nearby_levels(resistances, threshold_pct=0.01)

    supports.sort(key=lambda x: x["strength"], reverse=True)
    resistances.sort(key=lambda x: x["strength"], reverse=True)

    return {
        "current_price": current_price,
        "support_levels": supports,
        "resistance_levels": resistances,
        "key_support": supports[0]["price"] if supports else None,
        "key_resistance": resistances[0]["price"] if resistances else None,
    }


# ---------- agents ----------

def market_agent(state: State) -> State:
    result = get_market_data(state["ticker"])

    if "error" in result:
        state["market_data"] = {}
        return state

    stats = result["stats"]

    state["market_data"] = {
        "current_price": stats.get("current_price"),
        "date_range": stats.get("date_range"),
        "last_trading_date": result["price_data"].index[-1].strftime("%Y-%m-%d"),
        "volatility": stats.get("annualized_volatility"),
        "returns": stats.get("avg_daily_return"),
        "drawdown": stats.get("max_drawdown"),
        "sharpe": stats.get("sharpe_ratio"),
    }

    state["price_data"] = result.get("price_data")

    return state


def technical_agent(state: State) -> State:
    df = state.get("price_data")

    if df is None:
        state["technical_signals"] = {}
        return state

    result = get_technical_indicators(df)
    indicators = result.get("indicators", {})

    state["technical_signals"] = {
        "current_price": indicators.get("current_price"),
        "sma_20": indicators.get("sma_20"),
        "sma_50": indicators.get("sma_50"),
        "rsi": indicators.get("rsi"),
        "macd": indicators.get("macd"),
        "macd_signal": indicators.get("macd_signal"),
        "adx": indicators.get("adx"),
        "bullish_count": result.get("bullish_count", 0),
        "bearish_count": result.get("bearish_count", 0),
        "rsi_thresholds": result.get("rsi_thresholds", {}),
        "summary": result.get("signal_summary"),
    }

    state["technical_full"] = result

    return state


def options_agent(state: State) -> State:
    result = get_options_data(state["ticker"])

    if "error" in result:
        state["options_data"] = {
            "implied_volatility": "unavailable",
            "put_call_ratio": "unavailable",
            "iv_skew": "unavailable",
            "max_pain": "unavailable",
            "data_quality": "unavailable",
            "missing_fields": ["implied_volatility", "put_call_ratio", "iv_skew", "max_pain"],
        }
        return state

    raw = {
        "implied_volatility": result.get("atm_iv"),
        "put_call_ratio": result.get("put_call_ratio"),
        "iv_skew": result.get("iv_skew"),
        "max_pain": result.get("max_pain"),
    }
    missing_fields = [k for k, v in raw.items() if v in (None, "")]

    # Swap nulls for "unavailable" so downstream prompts don't silently drop them.
    cleaned = {k: (v if v not in (None, "") else "unavailable") for k, v in raw.items()}

    if not missing_fields:
        data_quality = "complete"
    elif len(missing_fields) < len(raw):
        data_quality = "partial"
    else:
        data_quality = "unavailable"

    cleaned["data_quality"] = data_quality
    cleaned["missing_fields"] = missing_fields

    state["options_data"] = cleaned
    state["options_full"] = result
    return state


# ---------- routing ----------

def data_quality_check(state: State) -> State:
    """Flag whether market data came back usable so the conditional edge can branch."""
    market = state.get("market_data", {})
    price_data = state.get("price_data")

    if not market or price_data is None:
        state["data_retry_count"] = state.get("data_retry_count", 0) + 1
        state["data_quality"] = "fail"
    else:
        state["data_quality"] = "pass"

    return state


def check_data_quality(state: State) -> str:
    """Retry the market fetch up to twice, then move on regardless."""
    quality = state.get("data_quality", "pass")
    retry_count = state.get("data_retry_count", 0)

    if quality == "fail" and retry_count < 2:
        return "retry"
    return "proceed"


def route_by_asset_type(state: State) -> str:
    """Stocks skip the options chain; anything option-related runs it."""
    asset_type = state.get("asset_type", "unknown")
    if asset_type == "stock":
        return "skip_options"
    return "run_options"
