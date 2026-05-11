from datetime import datetime
import json
import pandas as pd

# rule-based option context and data quality scoring
def compute_dte(expiration):
    """Days to expiration. Accepts 'YYYY-MM-DD' or natural-language dates."""
    if not expiration:
        return None
    formats = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            exp = datetime.strptime(str(expiration).strip(), fmt)
            return max((exp - datetime.now()).days, 0)
        except Exception:
            continue
    return None

def compute_option_position_context(state):
    """
    Compute moneyness context from current price, strike, and position.
    Returns {} if data is missing.
    """
    market = state.get("market_data", {}) or {}
    details = state.get("option_details", {}) or {}

    price = market.get("current_price")
    strike = details.get("strike")
    position = details.get("position")
    expiration = details.get("expiration")

    if price is None or strike is None or strike == 0:
        return {}

    try:
        price = float(price)
        strike = float(strike)
    except (TypeError, ValueError):
        return {}

    moneyness_pct = (price - strike) / strike

    # Position-aware moneyness label
    if abs(moneyness_pct) < 0.02:
        moneyness = "ATM"
    elif position in ("long_call", "short_call"):
        moneyness = "ITM" if price > strike else "OTM"
    elif position in ("long_put", "short_put"):
        moneyness = "ITM" if price < strike else "OTM"
    else:
        moneyness = "ITM for calls / OTM for puts" if price > strike else "OTM for calls / ITM for puts"

    dte = compute_dte(expiration)

    return {
        "underlying_price": round(price, 2),
        "strike": round(strike, 2),
        "moneyness_pct": round(moneyness_pct, 4),
        "moneyness": moneyness,
        "position": position,
        "dte": dte,
    }


def estimate_assignment_risk(position, moneyness_pct, dte):
    """
    Rule-based assignment risk for short option positions.
    For long positions, assignment risk does not apply.
    """
    if position not in ("short_call", "short_put"):
        return "N/A"
    if moneyness_pct is None or dte is None:
        return "unknown"

    # For short calls: positive moneyness_pct = ITM (bad)
    # For short puts: negative moneyness_pct = ITM (bad)
    itm_metric = moneyness_pct if position == "short_call" else -moneyness_pct

    if itm_metric > 0.05 and dte <= 7:
        return "high"
    if itm_metric > 0 and dte <= 14:
        return "moderate"
    if itm_metric > 0:
        return "low_to_moderate"
    return "low"


def compute_data_completeness(state):
    """
    Score 0.0 to 1.0 based on which expected data fields are present.
    Lower score = more missing data = decision agent should reduce confidence.
    """
    required = ["market_data", "technical_signals", "news_sentiment"]
    optional_for_options = ["implied_volatility", "iv_skew", "max_pain", "put_call_ratio"]

    score = 1.0
    missing = []

    for key in required:
        val = state.get(key)
        if not val:
            score -= 0.2
            missing.append(key)

    if state.get("asset_type") in ("option", "both"):
        options = state.get("options_data", {}) or {}
        for k in optional_for_options:
            v = options.get(k)
            if v in (None, "N/A", "unavailable", ""):
                score -= 0.1
                missing.append(f"options.{k}")

    return {
        "score": max(0.0, round(score, 2)),
        "missing_fields": missing,
    }

def risk_agent(state: State) -> State:
    tech = state.get("technical_full")
    options = state.get("options_full")

    if not tech:
        state["risk_score"] = 0.5
        return state

    weighted = get_weighted_signals(tech, options)

    composite = weighted.get("composite_score", 0)

    risk_score = 1 - (composite + 100) / 200

    state["risk_score"] = round(float(risk_score), 3)
    state["weighted_signals"] = weighted

    return state

def combine_signals(quant_score, news_score, risk):
    conflict = (quant_score * news_score < 0)

    if risk > 0.7:
        wq, wn = 0.8, 0.2
    elif abs(news_score) > 0.4:
        wq, wn = 0.5, 0.5
    else:
        wq, wn = 0.6, 0.4

    combined = wq * quant_score + wn * news_score
    if conflict:
        combined *= 0.7

    return combined, conflict

def signal_combiner(state: State) -> State:
    """
    Explicitly combine signals with defined weights BEFORE
    passing to LLM. LLM's job is to explain and adjust,
    not to decide from scratch.
    """
    quant = state.get("weighted_signals", {})
    news = state.get("news_sentiment", {})
    risk = state.get("risk_score", 0.5)
    asset_type = state.get("asset_type", "unknown")

    quant_score = quant.get("composite_score", 0) / 100  
    news_score = news.get("score", 0)

    if asset_type == "option":
        wq, wn = 0.75, 0.25
    elif risk > 0.7:
        wq, wn = 0.8, 0.2
    elif abs(news_score) > 0.4:
        wq, wn = 0.5, 0.5
    else:
        wq, wn = 0.6, 0.4

    combined = wq * quant_score + wn * news_score
    conflict = (quant_score * news_score < 0) and \
               (abs(quant_score) > 0.1 and abs(news_score) > 0.1)

    if conflict:
        combined *= 0.7

    if combined > 0.15:
        pre_action = "BUY"
    elif combined < -0.15:
        pre_action = "SELL"
    else:
        pre_action = "HOLD"

    state["combined_analysis"] = {
        "combined_score": round(combined, 3),
        "pre_action": pre_action,
        "weights": {"quant": wq, "news": wn},
        "conflict_detected": conflict,
        "quant_score": round(quant_score, 3),
        "news_score": round(news_score, 3),
    }

    return state

# decision agent
def decision_agent(state: State) -> State:
    quant = state.get("weighted_signals", {})
    risk = state.get("risk_score", 0.5)
    news = state.get("news_sentiment", {})
    tech = state.get("technical_signals", {})
    options = state.get("options_data", {})
    user = state.get("user_profile", {})
    market = state.get("market_data", {})
    asset_type = state.get("asset_type", "unknown")
    option_details = state.get("option_details", {})
    combined = state.get("combined_analysis", {})
    feedback = state.get("reflection_feedback", "")
    revision_note = ""
    if feedback:
        revision_note = f"""
IMPORTANT: Your previous recommendation was rejected.
Feedback: {feedback}
Please revise your recommendation to address this feedback.
"""

    #rule-based option context (moneyness, DTE, assignment risk)
    option_context = {}
    assignment_risk = "N/A"
    if asset_type in ("option", "both"):
        option_context = compute_option_position_context(state)
        if option_context:
            assignment_risk = estimate_assignment_risk(
                position=option_context.get("position"),
                moneyness_pct=option_context.get("moneyness_pct"),
                dte=option_context.get("dte"),
            )
    state["option_context"] = option_context
    state["assignment_risk"] = assignment_risk
    data_completeness = compute_data_completeness(state)
    state["data_completeness"] = data_completeness

    completeness_note = f"""
=== DATA COMPLETENESS ===
Score: {data_completeness['score']} (1.0 = complete, 0.0 = mostly missing)
Missing fields: {data_completeness['missing_fields'] if data_completeness['missing_fields'] else 'none'}
If score is below 0.7 you MUST lower the confidence value and explicitly mention
which fields were unavailable in your reasoning.
"""

    common_data = f"""
User risk profile: {user.get("risk_profile", "moderate")}

=== MARKET DATA ===
Current price: {market.get("current_price", "N/A")}
Date range: {market.get("date_range", "N/A")}
Last trading date: {market.get("last_trading_date", "N/A")}
Recent 5-day return: {market.get("recent_5d_return", "N/A")}
Recent 20-day return: {market.get("recent_20d_return", "N/A")}
Volatility: {market.get("volatility", "N/A")}
Sharpe ratio: {market.get("sharpe", "N/A")}
Max drawdown: {market.get("drawdown", "N/A")}

=== TECHNICAL INDICATORS ===
Current price from technical data: {tech.get("current_price", "N/A")}
SMA 20: {tech.get("sma_20", "N/A")}
SMA 50: {tech.get("sma_50", "N/A")}
RSI: {tech.get("rsi", "N/A")}
MACD: {tech.get("macd", "N/A")}
MACD signal: {tech.get("macd_signal", "N/A")}
ADX: {tech.get("adx", "N/A")}
Bullish signals: {tech.get("bullish_count", 0)} | Bearish: {tech.get("bearish_count", 0)}
RSI thresholds (adaptive): oversold<{tech.get("rsi_thresholds", {}).get("oversold", "N/A")}, overbought>{tech.get("rsi_thresholds", {}).get("overbought", "N/A")} ({tech.get("rsi_thresholds", {}).get("regime", "N/A")}, ann. vol={tech.get("rsi_thresholds", {}).get("annualized_vol", "N/A")})
Summary: {tech.get("summary", "N/A")}

=== NEWS SENTIMENT ===
Score: {news.get("score", "N/A")} (range: -1 to 1)
Summary: {news.get("summary", "N/A")}
Key events: {news.get("key_events", [])}
Content coverage: {news.get("content_coverage", "N/A")} (ratio of articles with full content, 0-1)

=== RISK AND COMBINED SIGNALS ===
Risk score: {risk} (0=low, 1=high)
Composite quant score: {quant.get("composite_score", "N/A")}
Quant confidence: {quant.get("confidence", "N/A")}
Recent momentum: {quant.get("recent_momentum", "N/A")}
Signal agreement: {quant.get("signal_agreement", "N/A")}
Recommendation strength: {quant.get("recommendation_strength", "N/A")}
Weighted direction: {quant.get("direction", "N/A")}

Combined score: {combined.get("combined_score", "N/A")} (range: -1 to 1)
Pre-action: {combined.get("pre_action", "N/A")}
Weights used: quant={combined.get("weights", {}).get("quant")}, news={combined.get("weights", {}).get("news")}
Signal conflict: {combined.get("conflict_detected", False)}
{completeness_note}
NOTE:
The pre-action is computed from quantitative rules.
You may adjust it, but you MUST explain the exact reason if your final action differs.

IMPORTANT REASONING STYLE:
Do NOT write the final reason as a list of indicators.
Use the data to form a trading thesis.
Your reasoning should explain:
1. The main reason for the action.
2. The strongest evidence supporting the action.
3. The strongest evidence against the action.
4. The execution plan or next step.

User question:
{state.get("query")}

{revision_note}
"""

    if asset_type == "stock":
        prompt = f"""You are a professional stock trading assistant.
The user is asking about STOCK trading, not options.

{common_data}

=== STOCK-SPECIFIC FOCUS ===
- Price context: Use current price and last trading date.
- Price trend: Is the stock showing actionable momentum or just mixed signals?
- Momentum: What do RSI and MACD imply?
- Trend strength: Does ADX confirm the move or show weak trend?
- Risk: Are volatility, drawdown, Sharpe ratio, and recent returns acceptable?
- News: Is news sentiment strong enough to support the trade?

INSTRUCTIONS:
1. Do NOT simply list every indicator.
2. Give a synthesized trading thesis.
3. Explain the main driver of the action.
4. Explain the strongest evidence against the action.
5. Use specific numbers only when they support the thesis.
6. If final action differs from pre-action, explain exactly why.
7. Give an execution plan: enter now, wait for pullback, position size, stop loss, or invalidation condition.
8. Adjust for the user risk profile.
9. If date and price are available, mention the data date and current price in the reason.

Return JSON:
{{
  "action": "BUY" or "SELL" or "HOLD",
  "thesis": "one-sentence main trading thesis",
  "reason": "synthesized reasoning, not a list of indicators",
  "key_drivers": ["main driver 1", "main driver 2", "main driver 3"],
  "conflicting_signals": ["opposing evidence or risk 1", "opposing evidence or risk 2"],
  "confidence": <float 0.0 to 1.0>,
  "entry_plan": "specific execution plan",
  "stop_loss": "suggested stop loss or N/A",
  "risk_warning": "key risks"
}}
"""

    elif asset_type == "option":
        if option_context:
            option_context_block = f"""
=== RULE-BASED OPTION CONTEXT (computed from option chain) ===
Underlying price: {option_context.get('underlying_price', 'N/A')}
Strike: {option_context.get('strike', 'N/A')}
Moneyness: {option_context.get('moneyness', 'N/A')} ({option_context.get('moneyness_pct', 'N/A'):+.2%} from strike)
Days to expiration (DTE): {option_context.get('dte', 'N/A')}
Position: {option_context.get('position', 'N/A')}
Assignment risk (rule-based): {assignment_risk}
"""
        else:
            option_context_block = """
=== RULE-BASED OPTION CONTEXT ===
Insufficient data to compute moneyness or DTE.
"""

        prompt = f"""You are a professional options trading assistant.
The user is asking about OPTIONS trading.

{common_data}

=== OPTIONS MARKET DATA ===
Put/Call ratio: {options.get("put_call_ratio", "N/A")}
Implied volatility (ATM): {options.get("implied_volatility", "N/A")}
IV skew: {options.get("iv_skew", "N/A")}
Max pain: {options.get("max_pain", "N/A")}
Data quality: {options.get("data_quality", "N/A")}
Missing fields: {options.get("missing_fields", [])}

=== USER'S POSITION ===
Strike: {option_details.get("strike", "N/A")}
Expiration: {option_details.get("expiration", "N/A")}
Position: {option_details.get("position", "N/A")}
{option_context_block}
=== OPTIONS-SPECIFIC FOCUS ===
- Trade structure: long call, short call, covered call, put, roll, or close.
- Moneyness (use the rule-based value above, not your own estimate).
- IV and skew: Is the option expensive, or is downside protection demand elevated?
- Time decay: Use the DTE value above directly.
- Assignment risk: Use the rule-based value above; do not contradict it.
- Max pain and put/call ratio: What do options-market signals imply?
- Technical indicators and news: Use them as supporting evidence, not the whole decision.

INSTRUCTIONS:
1. Do NOT simply list every indicator.
2. Focus on the option trade structure first.
3. Cite the moneyness, DTE, and assignment risk values explicitly in your reasoning.
4. Technical indicators and news are supporting evidence.
5. For existing short call or covered call positions, explicitly compare HOLD vs BUY BACK/CLOSE vs ROLL.
6. Explain why the rejected actions are worse than the selected action.
7. If final action differs from pre-action, explain exactly why.
8. Give a concrete execution plan and risk warning.
9. If options data is partial or unavailable, lower confidence and say so.
10. If date and price are available, mention the data date and current price in the reason.

Return JSON:
{{
  "action": "BUY" or "SELL" or "HOLD" or "ROLL" or "AVOID",
  "thesis": "one-sentence main option thesis",
  "reason": "synthesized reasoning focused on option risk/reward",
  "key_drivers": ["main driver 1", "main driver 2", "main driver 3"],
  "alternatives_considered": {{
    "hold": "why holding is good or bad",
    "close": "why closing is good or bad",
    "roll": "why rolling is good or bad"
  }},
  "confidence": <float 0.0 to 1.0>,
  "execution_plan": "specific next step",
  "theta_warning": "time decay assessment",
  "risk_warning": "key risks"
}}
"""
    else:
        prompt = f"""You are a professional trading assistant.

{common_data}

=== OPTIONS MARKET DATA ===
Put/Call ratio: {options.get("put_call_ratio", "N/A")}
Implied volatility: {options.get("implied_volatility", "N/A")}
IV skew: {options.get("iv_skew", "N/A")}
Max pain: {options.get("max_pain", "N/A")}

INSTRUCTIONS:
1. Do NOT simply list every indicator.
2. Form a trading thesis from the available market, technical, options, news, and risk data.
3. Explain the main driver of the action.
4. Explain the strongest evidence against the action.
5. When signals conflict, explain the conflict and justify your weighting.
6. If final action differs from pre-action, explain exactly why.
7. Give a concrete next step.
8. Adjust for user risk profile.
9. If date and price are available, mention the data date and current price in the reason.

Return JSON:
{{
  "action": "BUY" or "SELL" or "HOLD" or "ROLL" or "AVOID",
  "thesis": "one-sentence main trading thesis",
  "reason": "synthesized reasoning, not a list of indicators",
  "key_drivers": ["main driver 1", "main driver 2", "main driver 3"],
  "conflicting_signals": ["opposing evidence or risk 1", "opposing evidence or risk 2"],
  "confidence": <float 0.0 to 1.0>,
  "execution_plan": "specific next step",
  "risk_warning": "key risks"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)

    except Exception as e:
        result = {
            "action": "HOLD",
            "thesis": "Decision generation failed.",
            "reason": f"Decision failed: {str(e)}",
            "key_drivers": [],
            "conflicting_signals": [],
            "confidence": 0.0,
            "execution_plan": "N/A",
            "risk_warning": "System error"
        }

    state["decision"] = result
    state["llm_analysis"] = json.dumps(result, indent=2)

    return state

def reflection_agent(state: State) -> State:
    """
    Reviews the decision agent's output for quality and consistency.
    If the decision is poor, provides feedback for revision.
    Inspired by FinCon's self-critiquing mechanism.
    """
    decision = state.get("decision", {})
    tech = state.get("technical_signals", {})
    news = state.get("news_sentiment", {})
    quant = state.get("weighted_signals", {})
    options = state.get("options_data", {})

    prompt = f"""You are a senior risk manager reviewing a trading recommendation.

=== DECISION UNDER REVIEW ===
Action: {decision.get("action", "N/A")}
Reason: {decision.get("reason", "N/A")}
Confidence: {decision.get("confidence", "N/A")}

=== AVAILABLE DATA ===
Technical summary: {tech.get("summary", "N/A")}
Bullish: {tech.get("bullish_count", 0)} | Bearish: {tech.get("bearish_count", 0)}
News sentiment score: {news.get("score", "N/A")}
Composite quant score: {quant.get("composite_score", "N/A")}
Put/call ratio: {options.get("put_call_ratio", "N/A")}

=== REVIEW CRITERIA ===
1. Does the reasoning reference specific data points (numbers, not vague claims)?
2. Is there a contradiction between the action and the data?
   (e.g., recommending BUY when most signals are bearish)
3. Is the confidence level justified by signal agreement?
4. Does the recommendation include a risk warning?

Return JSON:
{{
  "approved": true or false,
  "issues": ["list of specific issues found, or empty if approved"],
  "feedback": "specific feedback for revision if not approved"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
    except Exception:
        result = {"approved": True, "issues": [], "feedback": ""}

    approved = result.get("approved", True)

    if not approved:
        state["reflection_count"] = state.get("reflection_count", 0) + 1
        state["reflection_feedback"] = result.get("feedback", "Please revise with more specific data references.")
    else:
        state["reflection_feedback"] = ""

    return state

def check_reflection(state: State) -> str:
    """
    After reflection: if decision was rejected and we haven't retried too many times,
    send back to decision agent for revision. Otherwise accept.
    """
    feedback = state.get("reflection_feedback", "")
    count = state.get("reflection_count", 0)

    if feedback and count <= 2:
        return "revise"
    return "accept"

# evaluation: weak baseline vs strong baseline vs multi-agent
# three-way comparison on 1-10 scale across five dimensions
def evaluate_response(query, response_text, system_name, available_signals=None):
    """LLM-as-judge scoring on five dimensions, 1-10 scale."""
    prompt = f"""
You are a strict evaluator of trading recommendation systems.
Be discriminating — do not award high scores easily.
The 9-10 range is reserved for rare, exceptional responses.

System: {system_name}

User Query:
{query}

System Response:
{response_text}

Available market signals:
{json.dumps(available_signals, indent=2, default=str) if available_signals else "N/A"}

Rate on these 5 dimensions from 1 to 10. Use the full range.

1. data_grounding (1-10)
Does the response cite specific numbers from the signals?
1  = no specific numbers at all
3  = a couple of numbers, mostly generic statements
5  = several numbers across one or two categories
7  = numbers spanning most relevant categories (price, technicals, options, news)
9  = numbers spanning all relevant categories AND includes derived metrics (e.g., moneyness pct, DTE in days, data completeness, sentiment method)
10 = all of the above AND tied to specific dates / contextualized vs thresholds

2. reasoning_completeness (1-10)
Does it cover the full set of relevant perspectives for the query type?
1  = one perspective only
3  = two perspectives, shallow
5  = three or four perspectives
7  = all required perspectives covered with adequate depth
9  = all perspectives covered AND explicitly weighs them against each other
10 = above AND explains alternatives that were rejected and why

3. consistency (1-10)
Does the recommendation follow logically from the cited data?
1  = recommendation contradicts the data
3  = partial alignment, with notable gaps
5  = generally aligned but glosses over conflicting signals
7  = aligned and acknowledges minor conflicts
9  = aligned, names the conflicts explicitly, and resolves them with stated weights
10 = above AND adjusts confidence to reflect remaining uncertainty

4. risk_awareness (1-10)
Does it engage seriously with risks?
1  = no risk discussion
3  = vague risk mention
5  = lists risks but does not quantify
7  = quantifies risks (volatility, drawdown, assignment risk, theta, gamma) using cited numbers
9  = above AND describes invalidation conditions (when would the recommendation flip?)
10 = above AND calibrates confidence based on data completeness or signal disagreement

5. actionability (1-10)
Is the recommendation specific enough to execute?
1  = vague, no concrete action
3  = direction stated (BUY/SELL/HOLD) but no plan
5  = direction + general timing
7  = direction + entry plan + stop loss / invalidation
9  = above AND alternative actions considered with explicit comparison
10 = above AND execution plan adapts to user risk profile

Return JSON only:
{{
  "data_grounding": <int 1-10>,
  "reasoning_completeness": <int 1-10>,
  "consistency": <int 1-10>,
  "risk_awareness": <int 1-10>,
  "actionability": <int 1-10>,
  "total": <sum of all scores, 5 to 50>,
  "explanation": "<brief explanation citing what kept it from 10s>"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "data_grounding": 0,
            "reasoning_completeness": 0,
            "consistency": 0,
            "risk_awareness": 0,
            "actionability": 0,
            "total": 0,
            "explanation": f"Evaluation failed: {str(e)}"
        }


def single_llm_baseline(query):
    """
    Weak baseline: single LLM call, no live data, no signals, no agents.
    Tests pure prior knowledge.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": query}],
        temperature=0
    )
    return response.choices[0].message.content


def tool_augmented_llm_baseline(query, available_signals):
    """
    Strong baseline: single LLM call, but with the SAME retrieved signals
    that the multi-agent system has access to. No routing, no reflection,
    no signal_combiner — just one prompt that asks the LLM to reason over
    everything at once.

    Any remaining gap with multi-agent reflects the value of routing,
    reflection, and explicit signal combination, not data access.
    """
    prompt = f"""You are a trading assistant.

User query:
{query}

You are given the same retrieved market signals as the multi-agent system.
These include market data, technical indicators, options-chain data,
FinBERT news sentiment, risk score, combined score, and (when applicable)
rule-based option context (moneyness, DTE, assignment risk).

SIGNALS:
{json.dumps(available_signals, indent=2, default=str)}

Give a clear recommendation (BUY / SELL / HOLD / ROLL / CLOSE / AVOID as
appropriate), a brief synthesized reason citing specific numbers from the
signals, and key risks. Do not just list every indicator — form a thesis.
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def make_initial_state(query):
    return {
        "query": query,
        "ticker": "",
        "task": "",
        "asset_type": "unknown",
        "option_details": {},
        "user_profile": {},
        "market_data": {},
        "technical_signals": {},
        "options_data": {},
        "price_data": None,
        "technical_full": {},
        "options_full": {},
        "news_sentiment": {},
        "llm_analysis": "",
        "risk_score": 0.0,
        "weighted_signals": {},
        "combined_analysis": {},
        "reflection_count": 0,
        "reflection_feedback": "",
        "data_retry_count": 0,
        "data_quality": "pass",
        "decision": {},
    }


def print_agent_decision_summary(agent_result):
    decision = agent_result.get("decision", {})

    print("\n--- Agent routing check ---")
    print("Ticker:", agent_result.get("ticker"))
    print("Asset type:", agent_result.get("asset_type"))
    print("Task:", agent_result.get("task"))
    print("Option details:", json.dumps(agent_result.get("option_details", {}), indent=2, default=str))

    print("\n--- Market data check ---")
    print(json.dumps(agent_result.get("market_data", {}), indent=2, default=str))

    print("\n--- Technical signals check ---")
    print(json.dumps(agent_result.get("technical_signals", {}), indent=2, default=str))

    print("\n--- Options data check ---")
    print(json.dumps(agent_result.get("options_data", {}), indent=2, default=str))

    print("\n--- Rule-based option context ---")
    print(json.dumps(agent_result.get("option_context", {}), indent=2, default=str))
    print("Assignment risk:", agent_result.get("assignment_risk", "N/A"))

    print("\n--- Data completeness ---")
    print(json.dumps(agent_result.get("data_completeness", {}), indent=2, default=str))

    news = agent_result.get("news_sentiment", {})
    print("\n--- News sentiment check ---")
    print("Score:", news.get("score", "N/A"))
    print("Score method:", news.get("score_method", "N/A"))
    print("Content coverage:", news.get("content_coverage", "N/A"))
    print("Articles used:", news.get("articles_used", "N/A"))
    print("Full text articles:", news.get("full_text_articles", "N/A"))
    print("Content sources:", json.dumps(news.get("content_sources", {}), indent=2, default=str))
    print("Summary:", news.get("summary", "N/A"))
    print("Key events:", news.get("key_events", []))
    print("Headlines analyzed:", len(news.get("headlines", [])))

    print("\n--- Weighted signals check ---")
    print(json.dumps(agent_result.get("weighted_signals", {}), indent=2, default=str))

    print("\n--- Combined analysis check ---")
    print(json.dumps(agent_result.get("combined_analysis", {}), indent=2, default=str))

    print("\n--- Final agent decision summary ---")
    print("Action:", decision.get("action", "N/A"))
    print("Thesis:", decision.get("thesis", "N/A"))
    print("Confidence:", decision.get("confidence", "N/A"))

    print("\nReason:")
    print(decision.get("reason", "N/A"))

    print("\nKey drivers:")
    key_drivers = decision.get("key_drivers", [])
    if isinstance(key_drivers, list) and key_drivers:
        for item in key_drivers:
            print("-", item)
    else:
        print("N/A")

    print("\nConflicting signals:")
    conflicting_signals = decision.get("conflicting_signals", [])
    if isinstance(conflicting_signals, list) and conflicting_signals:
        for item in conflicting_signals:
            print("-", item)
    else:
        print("N/A")

    print("\nEntry / execution plan:")
    print(decision.get("entry_plan", decision.get("execution_plan", "N/A")))

    if "alternatives_considered" in decision:
        print("\nAlternatives considered:")
        print(json.dumps(decision.get("alternatives_considered", {}), indent=2, default=str))

    if "theta_warning" in decision:
        print("\nTheta warning:")
        print(decision.get("theta_warning", "N/A"))

    print("\nStop loss:")
    print(decision.get("stop_loss", "N/A"))

    print("\nRisk warning:")
    print(decision.get("risk_warning", "N/A"))

test_queries = [
    """
    Should I buy TSLA stock right now?
    Use current market data, technical indicators, news sentiment, and risk.
    Give a clear BUY, SELL, or HOLD recommendation.
    """,

    """
    I want to buy COIN call options.

    Should I buy calls now, wait, or avoid the trade?

    Consider:
    - current price trend
    - RSI
    - MACD
    - ADX
    - implied volatility
    - IV skew
    - put/call ratio
    - max pain
    - news sentiment
    - risk-adjusted return
    """,

    """
    I sold a covered call on MSTR:
    - Strike: 180
    - Expiration: May 15, 2026
    - Position: short call
    - Premium received: 6.00

    Should I hold, buy back, or roll this option?

    Consider:
    - current stock price relative to strike
    - moneyness
    - implied volatility
    - max pain
    - put/call ratio
    - assignment risk
    - gamma risk near expiration
    - theta decay
    - technical indicators
    - news sentiment
    """
]

print(f"Prepared {len(test_queries)} volatile-stock test queries for evaluation.")

results = []
for i, query in enumerate(test_queries):
    print(f"\n{'=' * 70}")
    print(f"Query {i + 1}/{len(test_queries)}")
    print(query[:500])
    print("=" * 70)
    print("Running multi-agent system...")
    agent_result = graph.invoke(make_initial_state(query))

    decision = agent_result.get("decision", {})
    agent_response = json.dumps(decision, indent=2, default=str)
    news_result = agent_result.get("news_sentiment", {})

    available_signals = {
        "ticker": agent_result.get("ticker"),
        "task": agent_result.get("task"),
        "asset_type": agent_result.get("asset_type"),
        "option_details": agent_result.get("option_details", {}),
        "market": agent_result.get("market_data", {}),
        "technical": agent_result.get("technical_signals", {}),
        "options": agent_result.get("options_data", {}),
        "option_context": agent_result.get("option_context", {}),
        "assignment_risk": agent_result.get("assignment_risk", "N/A"),
        "data_completeness": agent_result.get("data_completeness", {}),
        "news": news_result,
        "risk_score": agent_result.get("risk_score"),
        "weighted": agent_result.get("weighted_signals", {}),
        "combined": agent_result.get("combined_analysis", {}),
    }

    print("Running weak baseline (single LLM, no signals)...")
    weak_response = single_llm_baseline(query)

    print("Running strong baseline (tool-augmented LLM, same signals)...")
    strong_response = tool_augmented_llm_baseline(query, available_signals)

    print("Evaluating weak baseline...")
    weak_eval = evaluate_response(query, weak_response, "Single LLM (weak baseline)", available_signals=None)

    print("Evaluating strong baseline...")
    strong_eval = evaluate_response(query, strong_response, "Tool-augmented LLM (strong baseline)", available_signals=available_signals)

    print("Evaluating multi-agent...")
    agent_eval = evaluate_response(query, agent_response, "Multi-Agent System", available_signals=available_signals)

    results.append({
        "query": query,
        "weak_response": weak_response,
        "strong_response": strong_response,
        "agent_response": agent_response,
        "weak_scores": weak_eval,
        "strong_scores": strong_eval,
        "agent_scores": agent_eval,
        "agent_revisions": agent_result.get("reflection_count", 0),
        "agent_asset_type": agent_result.get("asset_type", "unknown"),
        "agent_ticker": agent_result.get("ticker", "N/A"),
        "agent_task": agent_result.get("task", "N/A"),
        "agent_action": decision.get("action", "N/A"),
        "agent_thesis": decision.get("thesis", "N/A"),
        "agent_confidence": decision.get("confidence", "N/A"),
        "option_details": agent_result.get("option_details", {}),
        "options_data": agent_result.get("options_data", {}),
        "option_context": agent_result.get("option_context", {}),
        "assignment_risk": agent_result.get("assignment_risk", "N/A"),
        "data_completeness": agent_result.get("data_completeness", {}),
        "news_score": news_result.get("score", "N/A"),
        "news_method": news_result.get("score_method", "N/A"),
        "news_coverage": news_result.get("content_coverage", "N/A"),
        "news_key_events": news_result.get("key_events", []),
        "articles_used": news_result.get("articles_used", "N/A"),
        "full_text_articles": news_result.get("full_text_articles", "N/A"),
    })

    print_agent_decision_summary(agent_result)

    print("\n--- Scores (out of 50) ---")
    print(f"Weak baseline total:   {weak_eval.get('total', 0)}/50")
    print(f"Strong baseline total: {strong_eval.get('total', 0)}/50")
    print(f"Multi-agent total:     {agent_eval.get('total', 0)}/50")
    print(f"Agent action:   {decision.get('action', 'N/A')}")
    print(f"Agent thesis:   {decision.get('thesis', 'N/A')}")

dimensions = [
    "data_grounding",
    "reasoning_completeness",
    "consistency",
    "risk_awareness",
    "actionability"
]

weak_avgs = {}
strong_avgs = {}
agent_avgs = {}

for dim in dimensions:
    weak_avgs[dim] = sum(r["weak_scores"].get(dim, 0) for r in results) / len(results)
    strong_avgs[dim] = sum(r["strong_scores"].get(dim, 0) for r in results) / len(results)
    agent_avgs[dim] = sum(r["agent_scores"].get(dim, 0) for r in results) / len(results)

weak_avgs["average"] = sum(weak_avgs[d] for d in dimensions) / len(dimensions)
strong_avgs["average"] = sum(strong_avgs[d] for d in dimensions) / len(dimensions)
agent_avgs["average"] = sum(agent_avgs[d] for d in dimensions) / len(dimensions)

summary_df = pd.DataFrame({
    "Metric": [d.replace("_", " ").title() for d in dimensions] + ["Average"],
    "Weak Baseline": [f"{weak_avgs[d]:.1f}/10" for d in dimensions] + [f"{weak_avgs['average']:.1f}/10"],
    "Strong Baseline": [f"{strong_avgs[d]:.1f}/10" for d in dimensions] + [f"{strong_avgs['average']:.1f}/10"],
    "Multi-Agent": [f"{agent_avgs[d]:.1f}/10" for d in dimensions] + [f"{agent_avgs['average']:.1f}/10"],
    "Agent vs Strong": [f"{agent_avgs[d] - strong_avgs[d]:+.1f}" for d in dimensions] + [f"{agent_avgs['average'] - strong_avgs['average']:+.1f}"],
})

print("\n" + "=" * 70)
print("EVALUATION RESULTS (1-10 scale, three-way comparison)")
print("=" * 70)
print(summary_df.to_string(index=False))

detail_rows = []

for r in results:
    detail_rows.append({
        "Query": r["query"][:60].replace("\n", " ") + "...",
        "Ticker": r["agent_ticker"],
        "Task": r["agent_task"],
        "Type": r["agent_asset_type"],
        "Action": r["agent_action"],
        "Confidence": r["agent_confidence"],
        "DataCompl": r["data_completeness"].get("score", "N/A") if isinstance(r["data_completeness"], dict) else "N/A",
        "AssignRisk": r["assignment_risk"],
        "News Score": r["news_score"],
        "Thesis": str(r["agent_thesis"])[:60] + "...",
        "Weak": f"{r['weak_scores'].get('total', 0)}/50",
        "Strong": f"{r['strong_scores'].get('total', 0)}/50",
        "Agent": f"{r['agent_scores'].get('total', 0)}/50",
        "Revisions": r["agent_revisions"],
    })

detail_df = pd.DataFrame(detail_rows)

print("\n" + "=" * 70)
print("PER-QUERY SCORES (out of 50)")
print("=" * 70)
print(detail_df.to_string(index=False))

def win_rate(results, baseline_key):
    wins = sum(1 for r in results if r["agent_scores"].get("total", 0) > r[baseline_key].get("total", 0))
    ties = sum(1 for r in results if r["agent_scores"].get("total", 0) == r[baseline_key].get("total", 0))
    losses = len(results) - wins - ties
    return wins, ties, losses

w_w, w_t, w_l = win_rate(results, "weak_scores")
s_w, s_t, s_l = win_rate(results, "strong_scores")

print("\n" + "=" * 70)
print("WIN RATE")
print("=" * 70)
print(f"Multi-agent vs WEAK baseline:   W/T/L = {w_w}/{w_t}/{w_l}  ({w_w/len(results)*100:.0f}% win rate)")
print(f"Multi-agent vs STRONG baseline: W/T/L = {s_w}/{s_t}/{s_l}  ({s_w/len(results)*100:.0f}% win rate)")


news_rows = []

for r in results:
    news_rows.append({
        "Ticker": r["agent_ticker"],
        "News Score": r["news_score"],
        "News Method": r["news_method"],
        "Coverage": r["news_coverage"],
        "Articles Used": r["articles_used"],
        "Full Text Articles": r["full_text_articles"],
        "Key Events": "; ".join(r["news_key_events"][:2]) if isinstance(r["news_key_events"], list) else "N/A"
    })

news_df = pd.DataFrame(news_rows)

print("\n" + "=" * 70)
print("NEWS MODULE SUMMARY")
print("=" * 70)
print(news_df.to_string(index=False))

print("\n" + "=" * 70)
print("EVALUATOR EXPLANATIONS")
print("=" * 70)

for i, r in enumerate(results):
    print(f"\nQuery {i + 1}:")
    print("Weak baseline explanation:")
    print(r["weak_scores"].get("explanation", "N/A"))

    print("\nStrong baseline explanation:")
    print(r["strong_scores"].get("explanation", "N/A"))

    print("\nMulti-agent explanation:")
    print(r["agent_scores"].get("explanation", "N/A"))

try:
    summary_df.to_csv("evaluation_summary.csv", index=False)
    detail_df.to_csv("per_query_scores.csv", index=False)
    news_df.to_csv("news_module_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("CSV files saved:")
    print(" - evaluation_summary.csv")
    print(" - per_query_scores.csv")
    print(" - news_module_summary.csv")
    print("=" * 70)
except Exception as e:
    print(f"CSV save failed: {e}")
