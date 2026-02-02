import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple
import feedparser
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer

from config import CRYPTO_KEYWORDS, RSS_FEEDS, MAX_ITEMS_PER_FEED
import csv

# Only include articles published within the last N days
MAX_DAYS_OLD = 7  # Adjust this to change how far back to look


@dataclass
class ArticleRecord:
    source: str
    title: str
    link: str
    crypto: str
    compound: float  # VADER compound score
    pos: float
    neu: float
    neg: float
    published_date: str  # ISO date (YYYY-MM-DD) for time-weighted aggregation


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_cryptos(text: str) -> List[str]:
    """Return list of crypto names mentioned in the text based on keyword mapping."""
    text_norm = normalize_text(text)
    found = []
    for crypto, keywords in CRYPTO_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text_norm):
                found.append(crypto)
                break
    return found


def fetch_feed_items() -> List[Tuple[str, dict]]:
    """Fetch items from all configured RSS feeds.

    Returns list of (source_name, entry_dict).
    
    Note: We collect more items than MAX_ITEMS_PER_FEED to ensure we have
    coverage across all days in the date window (MAX_DAYS_OLD). The date
    filter in collect_articles() will handle limiting to recent articles.
    """
    items: List[Tuple[str, dict]] = []
    # Collect more items to ensure we cover all days in the date window
    # Most feeds won't have this many items, but this ensures we don't miss older days
    items_per_feed = max(MAX_ITEMS_PER_FEED * 3, 200)  # At least 200, or 3x the configured limit
    
    for source, urls in RSS_FEEDS.items():
        for url in urls:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:items_per_feed]:
                items.append((source, entry))
    return items


def extract_published_date(entry: dict) -> str:
    """
    Extract a best-effort publication date from an RSS/Atom entry and
    normalise it to an ISO date string (YYYY-MM-DD).

    If no date is available, fall back to today's UTC date so that
    the record can still participate in time-weighted aggregation.
    """
    dt = None

    # feedparser exposes *_parsed as time.struct_time when available
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed is not None:
            dt = datetime.utcfromtimestamp(datetime(*parsed[:6]).timestamp())
            break

    # Some feeds only expose text dates
    if dt is None:
        for key in ("published", "updated"):
            raw = entry.get(key)
            if raw:
                try:
                    # Very loose parse; most RSS dates follow RFC822-like formats
                    dt = datetime(*feedparser._parse_date(raw)[:6])  # type: ignore[attr-defined]
                    break
                except Exception:
                    continue

    if dt is None:
        dt = datetime.utcnow()

    return dt.date().isoformat()


def build_article_text(entry: dict) -> str:
    parts = []
    title = entry.get("title", "")
    summary = entry.get("summary", "") or entry.get("description", "")
    content = ""
    if "content" in entry and entry["content"]:
        # Some feeds provide richer HTML content
        content = " ".join(part.get("value", "") for part in entry["content"])
    parts.extend([title, summary, content])
    joined = " ".join(p for p in parts if p)
    joined = clean_html(joined)
    return joined


def collect_articles() -> List[ArticleRecord]:
    """Fetch items from feeds and return article records tagged with cryptocurrencies,
    including VADER sentiment scores.
    
    Only includes articles published within MAX_DAYS_OLD days.
    """
    records: List[ArticleRecord] = []
    sia = SentimentIntensityAnalyzer()
    
    # Calculate cutoff date
    cutoff_date = (datetime.utcnow() - timedelta(days=MAX_DAYS_OLD)).date()

    for source, entry in fetch_feed_items():
        published_date_str = extract_published_date(entry)
        try:
            published_date = datetime.strptime(published_date_str, "%Y-%m-%d").date()
        except ValueError:
            # If date parsing fails, skip this article
            continue
        
        # Skip articles older than MAX_DAYS_OLD
        if published_date < cutoff_date:
            continue
        
        text = build_article_text(entry)
        if not text:
            continue
        cryptos = detect_cryptos(text)
        if not cryptos:
            continue

        scores = sia.polarity_scores(text)
        for crypto in cryptos:
            article = ArticleRecord(
                source=source,
                title=entry.get("title", ""),
                link=entry.get("link", ""),
                crypto=crypto,
                compound=scores["compound"],
                pos=scores["pos"],
                neu=scores["neu"],
                neg=scores["neg"],
                published_date=published_date_str,
            )
            records.append(article)

    return records


def save_to_csv(records: List[ArticleRecord], path: str = "results.csv") -> None:
    """Save collected article records (with sentiment) to a CSV file."""
    fieldnames = [
        "source",
        "crypto",
        "title",
        "link",
        "compound",
        "pos",
        "neu",
        "neg",
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
                    "link": r.link,
                    "compound": r.compound,
                    "pos": r.pos,
                    "neu": r.neu,
                    "neg": r.neg,
                    "published_date": r.published_date,
                }
            )


def main() -> None:
    records = collect_articles()
    save_to_csv(records)
    print(f"\nSaved {len(records)} records to results.csv")


if __name__ == "__main__":
    main()

