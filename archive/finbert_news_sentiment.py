import os
import re
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import CRYPTO_KEYWORDS
from dotenv import load_dotenv

# Only include articles published within the last N days
MAX_DAYS_OLD = 7  # Adjust this to change how far back to look

# Minimum number of words required in article text (title + description)
MIN_WORD_COUNT = 10

# Minimum confidence threshold: filter articles where neutral probability is too high
# (i.e., we want articles with clear sentiment direction)
MIN_SENTIMENT_CONFIDENCE = 0.4  # Max neutral probability allowed

# Sentiment keywords that indicate strong positive or negative market sentiment
# Strong indicators get higher weight in adjustment
POSITIVE_KEYWORDS_STRONG = [
    "darling", "soar", "soared", "soars", "soaring", "bullish", "bull", "rally", "rallying",
    "surge", "surged", "surges", "surging", "explodes", "exploded", "explode", "explosion",
    "jump", "jumped", "jumps", "jumping", "skyrocket", "skyrocketed", "surpass", "surpassed",
    "breakthrough", "break out", "bull run", "record high", "all-time high", "ath", "new high",
    "momentum", "breakout", "breakouts", "supercycle", "super cycle", "institutional supercycle",
    "most bullish", "strongest signal", "builds strong case", "strong case", "outperformer",
    "outperform", "outperformed", "outperforming"
]

POSITIVE_KEYWORDS_MEDIUM = [
    "buy", "bought", "buying", "gain", "gained", "gains", "rising", "rise", "rose",
    "upward", "uptrend", "pump", "increase", "increased", "increasing", "grow", "grew", 
    "growing", "growth", "profit", "profitable", "profits", "support", "supportive",
    "adoption", "adopted", "institutional", "approve", "approved", "approval", 
    "partnership", "partnerships", "upgrade", "upgraded", "launch", "launched", 
    "success", "successful", "accumulate", "accumulation", "hold", "holding",
    "long", "longs", "optimistic", "optimism", "positive", "positivity", 
    "recovery", "recover", "rebound", "rebounded", "outflow", "inflow"  # Note: outflows can be negative context
]

NEGATIVE_KEYWORDS_STRONG = [
    "crash", "crashed", "crashes", "crashing", "bearish", "bear", "plunge", "plunged", 
    "plunges", "plunging", "dump", "dumping", "dumps", "dumped", "collapse", "collapsed",
    "collapses", "collapsing", "sell-off", "selloff", "bloodbath", "carnage", "disaster",
    "catastrophe", "failure", "failed", "failures", "panic", "panicked", "panicking",
    "reject", "rejected", "rejection", "ban", "banned", "banning", "fraud", "fraudulent",
    "scam", "scams", "hack", "hacked", "hacking", "security breach", "bubble", "burst"
]

NEGATIVE_KEYWORDS_MEDIUM = [
    "sell", "sold", "selling", "drop", "dropped", "drops", "dropping", "fall", "fell", 
    "falls", "falling", "decline", "declined", "declines", "declining", "down", "downward",
    "downtrend", "decrease", "decreased", "decreasing", "shrink", "shrunk", "shrinking",
    "loss", "losses", "losing", "resistance", "regulate", "regulated", "regulation",
    "regulatory", "investigation", "investigate", "volatility", "volatile", "uncertainty",
    "uncertain", "risk", "risky", "risks", "fear", "fearful", "concern", "concerns",
    "worried", "worry", "warning", "warnings", "short", "shorts", "shorting",
    "pessimistic", "pessimism", "negative", "negativity", "correction", "corrections",
    "bear market", "oversold", "profit-taking", "profit taking", "resistance level"
]

# Combine for detection
POSITIVE_KEYWORDS = POSITIVE_KEYWORDS_STRONG + POSITIVE_KEYWORDS_MEDIUM
NEGATIVE_KEYWORDS = NEGATIVE_KEYWORDS_STRONG + NEGATIVE_KEYWORDS_MEDIUM

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# Wall Street Journal RSS (markets / business headlines)
WSJ_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.a.dj.com/rss/RSSWSJD.xml",
]


@dataclass
class FinBertArticle:
    source: str
    title: str
    description: str
    url: str
    crypto: str
    label: str
    score: float
    prob_positive: float
    prob_neutral: float
    prob_negative: float
    published_date: str  # ISO date (YYYY-MM-DD) for time-weighted aggregation


def clean_html(text: str) -> str:
    """Clean HTML and remove URLs, special characters that don't help sentiment analysis."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ", strip=True)
    
    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    
    return cleaned.strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def count_words(text: str) -> int:
    """Count the number of words in a text string."""
    if not text:
        return 0
    return len(text.split())


def detect_sentiment_keywords(text: str) -> Tuple[float, float]:
    """Detect positive and negative sentiment keywords in text with weighted scoring.
    
    Strong keywords get 3x weight, medium keywords get 1x weight.
    Handles both single words and multi-word phrases.
    Returns (positive_score, negative_score) tuple where scores can be > 1.0.
    """
    text_norm = normalize_text(text)
    
    # First check for multi-word phrases (need to match as whole phrases)
    # Phrases like "builds strong case", "all-time high", "bull run"
    pos_strong = 0.0
    neg_strong = 0.0
    
    # Check phrases with spaces first (multi-word)
    phrases_strong_pos = [kw for kw in POSITIVE_KEYWORDS_STRONG if " " in kw or "-" in kw]
    phrases_strong_neg = [kw for kw in NEGATIVE_KEYWORDS_STRONG if " " in kw or "-" in kw]
    
    for phrase in phrases_strong_pos:
        if phrase.lower() in text_norm:
            pos_strong += 3.0
    
    for phrase in phrases_strong_neg:
        if phrase.lower() in text_norm:
            neg_strong += 3.0
    
    # Then check single-word keywords (using word boundaries for accuracy)
    single_pos_strong = [kw for kw in POSITIVE_KEYWORDS_STRONG if " " not in kw and "-" not in kw]
    single_neg_strong = [kw for kw in NEGATIVE_KEYWORDS_STRONG if " " not in kw and "-" not in kw]
    
    pos_strong += sum(3.0 for kw in single_pos_strong if re.search(rf"\b{re.escape(kw)}\b", text_norm))
    neg_strong += sum(3.0 for kw in single_neg_strong if re.search(rf"\b{re.escape(kw)}\b", text_norm))
    
    # Check medium keywords (single words)
    pos_medium = sum(1.0 for kw in POSITIVE_KEYWORDS_MEDIUM if re.search(rf"\b{re.escape(kw)}\b", text_norm))
    neg_medium = sum(1.0 for kw in NEGATIVE_KEYWORDS_MEDIUM if re.search(rf"\b{re.escape(kw)}\b", text_norm))
    
    positive_score = pos_strong + pos_medium
    negative_score = neg_strong + neg_medium
    
    # Special handling for common financial phrases that indicate sentiment
    # Check for positive financial phrases
    if re.search(r"\b(builds?\s+strong\s+case|strong\s+case\s+for|darling|catalyst|catalysts)\b", text_norm):
        positive_score += 6.0  # Very strong positive signal
    
    if re.search(r"\b(outperformer|outperform|outperforming|breakout\s+trade)\b", text_norm):
        positive_score += 4.0
    
    if re.search(r"\b(most\s+bullish|bullish\s+thing|supercycle|institutional\s+supercycle)\b", text_norm):
        positive_score += 6.0
    
    # Check for negative financial phrases  
    if re.search(r"\b(outflow|outflows|profit-taking|profit\s+taking|sell-side)\b", text_norm):
        negative_score += 3.0
    
    # If we find strong conflicting keywords, trust the stronger signal
    if pos_strong > 0 and neg_strong > 0:
        # If both strong, go with the one that appears more
        if pos_strong > neg_strong:
            negative_score *= 0.3  # Reduce negative influence
        elif neg_strong > pos_strong:
            positive_score *= 0.3  # Reduce positive influence
    
    return float(positive_score), float(negative_score)


def adjust_probabilities_with_keywords(
    prob_positive: float,
    prob_neutral: float,
    prob_negative: float,
    positive_keyword_score: float,
    negative_keyword_score: float,
) -> Tuple[float, float, float]:
    """Adjust FinBERT probabilities based on sentiment keyword detection.
    
    Uses a more aggressive approach: when keyword signals are strong, they can
    override FinBERT significantly to fix obvious misclassifications.
    
    Args:
        prob_positive: Original positive probability
        prob_neutral: Original neutral probability
        prob_negative: Original negative probability
        positive_keyword_score: Weighted positive keyword score (strong keywords = 3, medium = 1)
        negative_keyword_score: Weighted negative keyword score (strong keywords = 3, medium = 1)
        
    Returns:
        Adjusted (prob_positive, prob_neutral, prob_negative) tuple
    """
    if positive_keyword_score == 0 and negative_keyword_score == 0:
        return prob_positive, prob_neutral, prob_negative
    
    keyword_diff = positive_keyword_score - negative_keyword_score
    total_keyword_score = positive_keyword_score + negative_keyword_score
    
    if total_keyword_score == 0:
        return prob_positive, prob_neutral, prob_negative
    
    # Determine adjustment strength based on keyword score strength
    # Strong signals (score >= 3) can override FinBERT by 50-70%
    # Medium signals (score >= 1) can override by 20-40%
    max_score = max(positive_keyword_score, negative_keyword_score)
    
    if max_score >= 6:  # Very strong signal (2+ strong keywords or many medium)
        adjustment_strength = 0.70  # Override FinBERT by 70%
    elif max_score >= 3:  # Strong signal (1+ strong keyword)
        adjustment_strength = 0.50  # Override FinBERT by 50%
    elif max_score >= 1:  # Medium signal
        adjustment_strength = 0.30  # Adjust by 30%
    else:
        adjustment_strength = 0.15  # Weak signal, minimal adjustment
    
    # Calculate how much to shift based on keyword dominance
    keyword_dominance = abs(keyword_diff) / max(total_keyword_score, 1.0)
    
    # Apply adjustment: shift probabilities toward keyword-indicated direction
    if keyword_diff > 0:  # Positive keywords dominate
        # Increase positive probability, decrease negative
        shift_amount = adjustment_strength * keyword_dominance
        prob_positive_adjusted = min(1.0, prob_positive + shift_amount * (1.0 - prob_positive))
        prob_negative_adjusted = max(0.0, prob_negative * (1.0 - shift_amount))
        # Reduce neutral proportionally
        prob_neutral_adjusted = max(0.0, prob_neutral * (1.0 - shift_amount * 0.7))
    else:  # Negative keywords dominate
        # Increase negative probability, decrease positive
        shift_amount = adjustment_strength * keyword_dominance
        prob_negative_adjusted = min(1.0, prob_negative + shift_amount * (1.0 - prob_negative))
        prob_positive_adjusted = max(0.0, prob_positive * (1.0 - shift_amount))
        # Reduce neutral proportionally
        prob_neutral_adjusted = max(0.0, prob_neutral * (1.0 - shift_amount * 0.7))
    
    # Normalize to ensure probabilities sum to 1.0
    total = prob_positive_adjusted + prob_neutral_adjusted + prob_negative_adjusted
    if total > 0:
        prob_positive_adjusted /= total
        prob_neutral_adjusted /= total
        prob_negative_adjusted /= total
    
    return prob_positive_adjusted, prob_neutral_adjusted, prob_negative_adjusted


def detect_cryptos(text: str) -> List[str]:
    text_norm = normalize_text(text)
    found: List[str] = []
    for crypto, keywords in CRYPTO_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text_norm):
                found.append(crypto)
                break
    return found


def extract_newsapi_date(article: dict) -> str:
    """Extract YYYY-MM-DD from NewsAPI article (publishedAt)."""
    raw = article.get("publishedAt") or ""
    if raw:
        try:
            # Example format: 2024-01-01T12:34:56Z
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except Exception:
            pass
    return datetime.utcnow().date().isoformat()


def extract_wsj_date(entry: dict) -> str:
    """
    Extract a best-effort publication date from WSJ RSS entry and
    normalise to YYYY-MM-DD.
    """
    dt = None
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed is not None:
            dt = datetime.utcfromtimestamp(datetime(*parsed[:6]).timestamp())
            break

    if dt is None:
        for key in ("published", "updated"):
            raw = entry.get(key)
            if raw:
                try:
                    dt = datetime(*feedparser._parse_date(raw)[:6])  # type: ignore[attr-defined]
                    break
                except Exception:
                    continue

    if dt is None:
        dt = datetime.utcnow()

    return dt.date().isoformat()


def fetch_newsapi_articles(api_key: str, query: str, page_size: int = 100) -> List[dict]:
    """Fetch articles from NewsAPI for a given query string.
    
    Only fetches articles from the last MAX_DAYS_OLD days.
    
    Note: Uses page_size=100 (NewsAPI max) to ensure we collect articles
    across all days in the date window, not just the most recent ones.
    """
    # Calculate date range for NewsAPI
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=MAX_DAYS_OLD)
    
    params = {
        "q": query,
        "language": "en",
        "pageSize": min(page_size, 100),  # NewsAPI max is 100
        "sortBy": "publishedAt",
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
    }
    headers = {"X-Api-Key": api_key}
    resp = requests.get(NEWSAPI_ENDPOINT, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])


def fetch_wsj_articles() -> List[Tuple[str, dict]]:
    """Fetch articles from WSJ RSS feeds."""
    items: List[Tuple[str, dict]] = []
    for url in WSJ_FEEDS:
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            items.append(("Wall Street Journal", entry))
    return items


def load_finbert():
    """Load FinBERT model & tokenizer (cached globally)."""
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def finbert_predict(texts: List[str], tokenizer, model, apply_keyword_adjustment: bool = True) -> List[Tuple[str, float, float, float, float]]:
    """Run FinBERT on a batch of texts with optional keyword-based sentiment adjustment.

    Returns list of (label, score, prob_pos, prob_neu, prob_neg) for each text.
    score is defined as prob_pos - prob_neg in [-1, 1].
    
    Args:
        texts: List of text strings to analyze
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        apply_keyword_adjustment: If True, adjust probabilities based on sentiment keywords
    """
    results: List[Tuple[str, float, float, float, float]] = []
    if not texts:
        return results

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            prob_negative = float(probs[0])
            prob_neutral = float(probs[1])
            prob_positive = float(probs[2])
            
            # Apply keyword-based adjustment if enabled
            if apply_keyword_adjustment:
                pos_score, neg_score = detect_sentiment_keywords(text)
                prob_positive, prob_neutral, prob_negative = adjust_probabilities_with_keywords(
                    prob_positive, prob_neutral, prob_negative, pos_score, neg_score
                )
            
            score = prob_positive - prob_negative
            # Determine label based on adjusted probabilities
            if prob_positive > prob_neutral and prob_positive > prob_negative:
                label = "positive"
            elif prob_negative > prob_neutral and prob_negative > prob_positive:
                label = "negative"
            else:
                label = "neutral"
            
            results.append(
                (label, score, prob_positive, prob_neutral, prob_negative)
            )
    return results


def remove_duplicate_articles(records: List[FinBertArticle]) -> List[FinBertArticle]:
    """Remove duplicate articles based on URL.
    
    If multiple records have the same URL and crypto, keep only the first one.
    """
    seen = set()
    unique_records: List[FinBertArticle] = []
    for record in records:
        key = (record.url, record.crypto)
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
    return unique_records


def collect_finbert_articles() -> List[FinBertArticle]:
    """Collect NewsAPI + WSJ articles, tag with cryptos, and compute FinBERT scores.
    
    Only includes articles published within MAX_DAYS_OLD days.
    Filters out:
    - Articles with text shorter than MIN_WORD_COUNT words
    - Duplicate articles (same URL and crypto)
    - Articles with low sentiment confidence (neutral probability > MIN_SENTIMENT_CONFIDENCE)
    """
    # Load environment variables from a .env file if present
    load_dotenv()

    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY environment variable is not set.")

    # Build a combined query from crypto names
    main_names = list(CRYPTO_KEYWORDS.keys())
    query = " OR ".join(main_names)

    tokenizer, model = load_finbert()

    records: List[FinBertArticle] = []
    
    # Calculate cutoff date for filtering
    cutoff_date = (datetime.utcnow() - timedelta(days=MAX_DAYS_OLD)).date()

    # 1) NewsAPI articles
    # Use max page size (100) to ensure we get articles across all days in the window
    articles = fetch_newsapi_articles(api_key, query=query, page_size=100)
    news_texts: List[str] = []
    news_meta: List[Tuple[str, str, str, str]] = []

    for a in articles:
        title = a.get("title", "") or ""
        description = a.get("description", "") or ""
        url = a.get("url", "") or ""
        published_date_str = extract_newsapi_date(a)
        
        # Double-check date filtering (in case NewsAPI returns older articles)
        try:
            published_date = datetime.strptime(published_date_str, "%Y-%m-%d").date()
            if published_date < cutoff_date:
                continue
        except ValueError:
            continue
        
        # Clean title and description separately
        title_clean = clean_html(title)
        description_clean = clean_html(description)
        
        # Weight title more heavily by including it twice (titles often contain key sentiment)
        # Format: "TITLE. TITLE. DESCRIPTION" - this gives title 2x weight in analysis
        combined = f"{title_clean}. {title_clean}. {description_clean}".strip()
        if not combined or not combined.strip():
            continue
        
        # Filter out articles that are too short (check original combined, not weighted)
        original_combined = f"{title_clean}. {description_clean}".strip()
        word_count = count_words(original_combined)
        if word_count < MIN_WORD_COUNT:
            continue
        
        cryptos = detect_cryptos(combined)
        if not cryptos:
            continue
        news_texts.append(combined)
        news_meta.append((title, description, url, published_date_str))

    news_scores = finbert_predict(news_texts, tokenizer, model, apply_keyword_adjustment=True)

    for (title, description, url, published_date), (
        label,
        score,
        p_pos,
        p_neu,
        p_neg,
    ) in zip(news_meta, news_scores):
        # Filter out articles with low sentiment confidence (too neutral)
        if p_neu > MIN_SENTIMENT_CONFIDENCE:
            continue
        
        combined = clean_html(f"{title}. {description}")
        cryptos = detect_cryptos(combined)
        for crypto in cryptos:
            records.append(
                FinBertArticle(
                    source="NewsAPI",
                    title=title,
                    description=description,
                    url=url,
                    crypto=crypto,
                    label=label,
                    score=score,
                    prob_positive=p_pos,
                    prob_neutral=p_neu,
                    prob_negative=p_neg,
                    published_date=published_date,
                )
            )

    # 2) WSJ RSS articles
    wsj_items = fetch_wsj_articles()
    wsj_texts: List[str] = []
    wsj_meta: List[Tuple[str, str, str, str]] = []

    for source, entry in wsj_items:
        title = entry.get("title", "") or ""
        summary = entry.get("summary", "") or entry.get("description", "") or ""
        url = entry.get("link", "") or ""
        published_date_str = extract_wsj_date(entry)
        
        # Filter by date
        try:
            published_date = datetime.strptime(published_date_str, "%Y-%m-%d").date()
            if published_date < cutoff_date:
                continue
        except ValueError:
            continue
        
        # Clean title and summary separately
        title_clean = clean_html(title)
        summary_clean = clean_html(summary)
        
        # Weight title more heavily by including it twice (titles often contain key sentiment)
        # Format: "TITLE. TITLE. SUMMARY" - this gives title 2x weight in analysis
        combined = f"{title_clean}. {title_clean}. {summary_clean}".strip()
        if not combined or not combined.strip():
            continue
        
        # Filter out articles that are too short (check original combined, not weighted)
        original_combined = f"{title_clean}. {summary_clean}".strip()
        word_count = count_words(original_combined)
        if word_count < MIN_WORD_COUNT:
            continue
        
        cryptos = detect_cryptos(combined)
        if not cryptos:
            continue
        wsj_texts.append(combined)
        wsj_meta.append((title, summary, url, published_date_str))

    wsj_scores = finbert_predict(wsj_texts, tokenizer, model, apply_keyword_adjustment=True)

    for (title, summary, url, published_date), (
        label,
        score,
        p_pos,
        p_neu,
        p_neg,
    ) in zip(wsj_meta, wsj_scores):
        # Filter out articles with low sentiment confidence (too neutral)
        if p_neu > MIN_SENTIMENT_CONFIDENCE:
            continue
        
        combined = clean_html(f"{title}. {summary}")
        cryptos = detect_cryptos(combined)
        for crypto in cryptos:
            records.append(
                FinBertArticle(
                    source="Wall Street Journal",
                    title=title,
                    description=summary,
                    url=url,
                    crypto=crypto,
                    label=label,
                    score=score,
                    prob_positive=p_pos,
                    prob_neutral=p_neu,
                    prob_negative=p_neg,
                    published_date=published_date,
                )
            )

    # Remove duplicate articles (same URL and crypto)
    records = remove_duplicate_articles(records)
    
    return records


def save_finbert_to_csv(records: List[FinBertArticle], path: str = "finbert_news_results.csv") -> None:
    import csv

    fieldnames = [
        "source",
        "crypto",
        "title",
        "description",
        "url",
        "label",
        "score",
        "prob_positive",
        "prob_neutral",
        "prob_negative",
        "published_date",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "source": r.source,
                    "crypto": r.crypto,
                    "title": r.title,
                    "description": r.description,
                    "url": r.url,
                    "label": r.label,
                    "score": r.score,
                    "prob_positive": r.prob_positive,
                    "prob_neutral": r.prob_neutral,
                    "prob_negative": r.prob_negative,
                    "published_date": r.published_date,
                }
            )


def main() -> None:
    records = collect_finbert_articles()
    save_finbert_to_csv(records)
    print(f"Saved {len(records)} FinBERT-scored articles to finbert_news_results.csv")


if __name__ == "__main__":
    main()

