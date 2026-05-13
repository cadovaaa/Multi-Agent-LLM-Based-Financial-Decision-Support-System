import json
import feedparser
import yfinance as yf
import urllib.parse
import httpx
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from google.colab import userdata
from openai import OpenAI
from typing import TypedDict, Dict, Any, Optional


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

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
client = OpenAI()

USER_MEMORY = {
    "default_user": {
        "risk_profile": "moderate",
        "history": []
    }
}

def get_user_profile(user_id="default_user"):
    return USER_MEMORY.get(user_id, USER_MEMORY["default_user"])

# Planner agent
def planner(state: State) -> State:
    query = state.get("query", "")
    user_id = state.get("user_id", "default_user")

    state["user_profile"] = get_user_profile(user_id)

    try:
        prompt = f"""
You are a trading assistant.

Extract structured information from the query.

Query:
"{query}"

Return JSON:
{{
  "ticker": "stock ticker if mentioned, else null",
  "task": one of [
    "buy_stock",
    "sell_stock",
    "buy_option",
    "sell_option",
    "hold_option",
    "close_option",
    "roll_option",
    "option_management",
    "general"
  ],
  "asset_type": one of ["stock", "option", "both", "unknown"],
  "option_details": {{
    "strike": <number or null>,
    "expiration": "<date or null>",
    "position": "long_call" or "long_put" or "short_call" or "short_put" or null
  }}
}}

Classification hints:
- If the query asks whether to hold, close, buy back, or roll an existing
  option position, classify task as "option_management" or "roll_option".
- If the query is about opening a new option trade, use "buy_option" or "sell_option".
- If the query is about a stock purchase or sale, use "buy_stock" or "sell_stock".
"""
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)

    except Exception:
        result = {"ticker": None, "task": "general",
                  "asset_type": "unknown", "option_details": {}}

    ticker = result.get("ticker")
    if ticker:
        state["ticker"] = ticker.upper()

    state["task"] = result.get("task", "general")
    state["asset_type"] = result.get("asset_type", "unknown")
    state["option_details"] = result.get("option_details", {})

    state.setdefault("market_data", {})
    state.setdefault("technical_signals", {})
    state.setdefault("options_data", {})
    state.setdefault("price_data", None)
    state.setdefault("technical_full", {})
    state.setdefault("options_full", {})
    state.setdefault("news_sentiment", {})
    state.setdefault("risk_score", 0.0)
    state.setdefault("weighted_signals", {})
    state.setdefault("llm_analysis", "")
    state.setdefault("decision", {})
    state.setdefault("reflection_count", 0)
    state.setdefault("reflection_feedback", "")
    state.setdefault("data_retry_count", 0)

    return state


# News sentiment agent
# FinBERT scores sentiment numerically; LLM is used only for summarization
try:
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None
    )
    FINBERT_AVAILABLE = True
    FINBERT_ERROR = None
except Exception as e:
    finbert = None
    FINBERT_AVAILABLE = False
    FINBERT_ERROR = str(e)
    print("FinBERT failed to load. Will use LLM fallback for sentiment score.")
    print(FINBERT_ERROR)


PAYWALLED_DOMAINS = [
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "barrons.com",
    "marketwatch.com"
]


def is_paywalled(url: str) -> bool:
    """
    Check whether the URL appears to be from a commonly paywalled source.
    """
    if not url:
        return False
    return any(domain in url.lower() for domain in PAYWALLED_DOMAINS)


def fetch_article_content(url: str, timeout: int = 5, max_chars: int = 1500) -> str:
    """
    Try to fetch article body text.

    If the content is unavailable, blocked, too short, or extraction fails,
    return an empty string.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        resp = httpx.get(
            url,
            timeout=timeout,
            headers=headers,
            follow_redirects=True
        )

        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
            tag.decompose()

        article = soup.find("article") or soup.find("main")

        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            paragraphs = soup.find_all("p")
            text = " ".join(
                p.get_text(separator=" ", strip=True)
                for p in paragraphs
            )

        text = " ".join(text.split())

        if len(text) < 200:
            return ""

        return text[:max_chars]

    except Exception as e:
        print(f"Article extraction failed for {url}: {e}")
        return ""


def finbert_score(texts: list) -> float:
    """
    Score a list of article texts or headlines using FinBERT.

    Output range:
    - positive close to +1
    - negative close to -1
    - neutral close to 0

    Formula:
    score = P(positive) - P(negative)

    This function is written to handle different transformers return formats:
    1. [[{"label": "...", "score": ...}, ...]]
    2. [{"label": "...", "score": ...}, ...]
    3. {"label": "...", "score": ...}
    """
    if not texts:
        return 0.0

    if not FINBERT_AVAILABLE:
        return None

    scores = []

    for text in texts:
        if not text or not isinstance(text, str):
            continue

        try:
            raw = finbert(text[:512])

            # Case 1 or 2: list output
            if isinstance(raw, list) and len(raw) > 0:
                result = raw[0]

                # Case 1: raw = [[{...}, {...}, {...}]]
                if isinstance(result, list):
                    label_scores = result

                # Case 2: raw = [{...}, {...}, {...}]
                elif isinstance(result, dict):
                    label_scores = raw

                else:
                    print("Unexpected FinBERT nested output:", raw)
                    continue

            # Case 3: dict output
            elif isinstance(raw, dict):
                label_scores = [raw]

            else:
                print("Unexpected FinBERT output:", raw)
                continue

            score_map = {
                item["label"].lower(): item["score"]
                for item in label_scores
                if isinstance(item, dict)
                and "label" in item
                and "score" in item
            }

            positive = score_map.get("positive", 0.0)
            negative = score_map.get("negative", 0.0)

            net_score = positive - negative
            scores.append(net_score)

        except Exception as e:
            print(f"FinBERT failed on one text: {e}")
            continue

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 3)


def news_agent(state: State) -> State:
    """
    News Sentiment Agent.

    Steps:
    1. Fetch recent news from Google News RSS.
    2. If Google News fails, fallback to yfinance news.
    3. Use RSS summary or article content when available.
    4. If unavailable, fallback to title only.
    5. Use FinBERT for sentiment scoring when available.
    6. Use LLM only for summary and key events.
    """
    ticker = state.get("ticker", "UNKNOWN")

    if ticker == "UNKNOWN" or ticker == "":
        state["news_sentiment"] = {
            "score": 0.0,
            "summary": "No ticker provided, cannot fetch news.",
            "headlines": [],
            "key_events": [],
            "content_coverage": 0.0,
            "score_method": "No ticker",
            "articles_used": 0,
            "full_text_articles": 0,
        }
        return state

    articles = []

    try:
        query = urllib.parse.quote(f"{ticker} stock")
        rss_url = (
            f"https://news.google.com/rss/search?q={query}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )

        feed = feedparser.parse(rss_url)

        if hasattr(feed, "entries") and len(feed.entries) > 0:
            for entry in feed.entries[:8]:
                title = entry.get("title", "")
                url = entry.get("link", "")
                published = entry.get("published", "")

                article = {
                    "title": title,
                    "url": url,
                    "published": published,
                    "content": "",
                    "content_source": "none"
                }

                # Google News RSS often includes summary snippets.
                rss_summary = entry.get("summary", "")
                if rss_summary:
                    soup = BeautifulSoup(rss_summary, "html.parser")
                    summary_text = soup.get_text(separator=" ", strip=True)
                    if len(summary_text) >= 50:
                        article["content"] = summary_text[:1500]
                        article["content_source"] = "rss_summary"

                # If RSS summary is unavailable, try article extraction.
                if (
                    not article["content"]
                    and article["url"]
                    and not is_paywalled(article["url"])
                ):
                    extracted_text = fetch_article_content(article["url"])
                    if extracted_text:
                        article["content"] = extracted_text
                        article["content_source"] = "article_extraction"

                articles.append(article)

    except Exception as e:
        print(f"Google News RSS failed: {e}")

    if not articles:
        try:
            stock = yf.Ticker(ticker)
            yf_news = stock.news

            for item in yf_news[:8]:
                title = item.get("title", "")
                url = item.get("link", "") or item.get("url", "")
                published = item.get("providerPublishTime", "")

                article = {
                    "title": title,
                    "url": url,
                    "published": published,
                    "content": "",
                    "content_source": "none"
                }

                if url and not is_paywalled(url):
                    extracted_text = fetch_article_content(url)
                    if extracted_text:
                        article["content"] = extracted_text
                        article["content_source"] = "article_extraction"

                articles.append(article)

        except Exception as e:
            print(f"yfinance news failed: {e}")

    if not articles:
        state["news_sentiment"] = {
            "score": 0.0,
            "summary": f"No recent news found for {ticker}. News source may be unavailable.",
            "headlines": [],
            "key_events": [],
            "content_coverage": 0.0,
            "score_method": "No news found",
            "articles_used": 0,
            "full_text_articles": 0,
        }
        return state

    headlines_only = []
    news_input = ""
    finbert_inputs = []

    for i, art in enumerate(articles):
        title = art.get("title", "")
        content = art.get("content", "")
        source = art.get("content_source", "none")

        headlines_only.append(title)

        if content:
            finbert_inputs.append(content[:512])
        else:
            finbert_inputs.append(title[:512])

        news_input += f"\n[Article {i + 1}]\n"
        news_input += f"Title: {title}\n"

        if content:
            news_input += f"Content source: {source}\n"
            news_input += f"Content excerpt: {content[:800]}\n"
        else:
            news_input += "Content source: title_only_fallback\n"
            news_input += "Content: unavailable; using title only.\n"

    content_coverage = round(
        sum(1 for art in articles if art.get("content")) / len(articles),
        3
    )

    full_text_articles = sum(
        1 for art in articles
        if art.get("content_source") == "article_extraction"
    )

    score = finbert_score(finbert_inputs)
    use_llm_score_fallback = score is None

    if use_llm_score_fallback:
        prompt = f"""
Analyze these financial news articles for {ticker}.

FinBERT was unavailable, so you should provide:
1. A sentiment score from -1.0 bearish to 1.0 bullish
2. A 3-4 sentence summary of the key fundamental events
3. The most important discrete events

Content coverage: {content_coverage}

News:
{news_input}

Return JSON:
{{
    "score": <float from -1.0 bearish to 1.0 bullish>,
    "summary": "<3-4 sentence summary focusing on key fundamental events>",
    "key_events": ["most important event 1", "most important event 2"]
}}
"""
    else:
        prompt = f"""
Analyze these financial news articles for {ticker}.

Your job is NOT to score sentiment.
FinBERT already produced the numerical sentiment score.

FinBERT sentiment score: {score}
Content coverage: {content_coverage}

Your job is to:
1. Summarize the key fundamental events in 3-4 sentences
2. Extract the most important discrete events
3. Mention whether the evidence is mostly based on RSS summaries, extracted article text, or title-only fallback.

News:
{news_input}

Return JSON:
{{
    "summary": "<3-4 sentence summary focusing on key fundamental events>",
    "key_events": ["most important event 1", "most important event 2"]
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

        if use_llm_score_fallback:
            score = float(result.get("score", 0.0))

        summary = result.get("summary", "")
        key_events = result.get("key_events", [])

    except Exception as e:
        if use_llm_score_fallback:
            score = 0.0

        summary = f"News summary failed: {str(e)}"
        key_events = []

    state["news_sentiment"] = {
        "score": round(float(score), 3),
        "summary": summary,
        "headlines": headlines_only,
        "key_events": key_events,
        "content_coverage": content_coverage,
        "score_method": (
            "FinBERT on RSS/article content with headline fallback"
            if not use_llm_score_fallback
            else "LLM fallback because FinBERT unavailable"
        ),
        "articles_used": len(articles),
        "full_text_articles": full_text_articles,
        "content_sources": {
            "rss_summary": sum(
                1 for art in articles
                if art.get("content_source") == "rss_summary"
            ),
            "article_extraction": sum(
                1 for art in articles
                if art.get("content_source") == "article_extraction"
            ),
            "title_only": sum(
                1 for art in articles
                if not art.get("content")
            ),
        },
        "article_sources": [
            {
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "content_source": art.get("content_source", "none")
            }
            for art in articles
        ]
    }

    return state



# Backtesting
def backtest_strategy(
    ticker: str,
    start="2020-01-01",
    end="2025-01-01",
    initial_capital=10000,
    transaction_cost=0.001
):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    LOOKBACK = 50
    signals = []
    combined_scores = []
    risk_scores = []

    for i in range(LOOKBACK, len(df)):
        historical_df = df.iloc[:i].copy()

        indicators = get_technical_indicators(historical_df)
        weighted = get_weighted_signals(indicators)

        # Original technical composite score: roughly -100 to 100
        quant_score = weighted.get("composite_score", 0) / 100

        # Historical backtest cannot safely use real historical news/options snapshots.
        # So we set news_score = 0 to avoid look-ahead bias.
        news_score = 0.0

        # Same risk-score idea as the full system:
        # risk = 1 - (composite_score + 100) / 200
        composite_raw = weighted.get("composite_score", 0)
        risk = 1 - (composite_raw + 100) / 200

        # Reduced full-system signal combiner logic
        # Default stock weighting: quant 0.6, news 0.4
        # If risk is high, use more quant/risk-aware weighting
        if risk > 0.7:
            wq, wn = 0.8, 0.2
        elif abs(news_score) > 0.4:
            wq, wn = 0.5, 0.5
        else:
            wq, wn = 0.6, 0.4

        combined = wq * quant_score + wn * news_score

        # Conflict penalty only applies if quant and news disagree meaningfully.
        # Since news_score = 0 here, conflict will normally be False.
        conflict = (quant_score * news_score < 0) and \
                   (abs(quant_score) > 0.1 and abs(news_score) > 0.1)

        if conflict:
            combined *= 0.7

        # Same BUY / SELL / HOLD threshold as signal_combiner
        if combined > 0.15:
            signal = 1      # long
        elif combined < -0.15:
            signal = 0      # out of market, simplified from SELL
        else:
            signal = 0      # HOLD / no new long position

        signals.append(signal)
        combined_scores.append(combined)
        risk_scores.append(risk)

    df = df.iloc[-len(signals):].copy()
    df["signal"] = signals
    df["combined_score"] = combined_scores
    df["risk_score"] = risk_scores

    df["position"] = df["signal"].shift(1).fillna(0)

    df["market_return"] = df["Close"].pct_change().fillna(0)
    df["trade"]         = df["position"].diff().abs().fillna(0)
    df["cost"]          = df["trade"] * transaction_cost
    df["strategy_return"] = (
        df["position"] * df["market_return"] - df["cost"]
    ).fillna(0)

    df["portfolio_value"]  = (1 + df["strategy_return"]).cumprod() * initial_capital
    df["buyhold_value"]    = (1 + df["market_return"]).cumprod()   * initial_capital

    def sharpe(returns):
        return (returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0 else 0.0)

    def max_dd(values):
        return (values / values.cummax() - 1).min()

    strat_ret  = df["portfolio_value"].iloc[-1] / initial_capital - 1
    bh_ret     = df["buyhold_value"].iloc[-1]   / initial_capital - 1
    strat_sh   = sharpe(df["strategy_return"])
    bh_sh      = sharpe(df["market_return"])
    strat_dd   = max_dd(df["portfolio_value"])
    bh_dd      = max_dd(df["buyhold_value"])
    n_trades   = int(df["trade"].sum())
    pct_long   = (df["position"] == 1).mean()

    metrics = {
        "ticker":            ticker,
        "strategy_return":   strat_ret,
        "buyhold_return":    bh_ret,
        "strategy_sharpe":   strat_sh,
        "buyhold_sharpe":    bh_sh,
        "strategy_maxdd":    strat_dd,
        "buyhold_maxdd":     bh_dd,
        "n_trades":          n_trades,
        "pct_long":          pct_long,
        "avg_combined_score": df["combined_score"].mean(),
        "avg_risk_score":     df["risk_score"].mean(),
    }

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2%}" if "return" in k or "dd" in k or "pct" in k
                  else f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    return df, metrics

# Run across tickers
tickers = ["NVDA", "TSLA", "COIN"]
all_metrics = []

fig, axes = plt.subplots(1, len(tickers), figsize=(15, 4), sharey=False)

for ax, ticker in zip(axes, tickers):
    df, m = backtest_strategy(ticker)
    all_metrics.append(m)

    ax.plot(df.index, df["portfolio_value"], label="Reduced Full-System Strategy")
    ax.plot(df.index, df["buyhold_value"], label="Buy & Hold", linestyle="--")
    ax.set_title(ticker)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8)

plt.suptitle("Reduced Full-System Backtest vs Buy & Hold (2020–2025)", y=1.02)
plt.tight_layout()
plt.show()
