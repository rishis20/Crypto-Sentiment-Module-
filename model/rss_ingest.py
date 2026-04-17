"""
RSS ingestion for Ollama-only sentiment pipeline.

This module collects crypto-related RSS entries and exports a CSV that can be
scored by analyze.py. It does not run any sentiment model itself.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import csv
import json
import re
import time
import xml.etree.ElementTree as ET

import feedparser
import requests
from bs4 import BeautifulSoup

from config import CRYPTO_KEYWORDS, RSS_FEEDS, MAX_ITEMS_PER_FEED

MAX_DAYS_OLD = 7


@dataclass
class RSSArticleRecord:
    source: str
    title: str
    summary: str
    url: str
    published_date: str
    crypto: str
    text_for_sentiment: str


@dataclass
class RawRSSRecord:
    source: str
    url: str
    published_at: str
    fetched_at: str
    title_raw: str
    summary_raw: str
    raw_payload: dict


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ", strip=True)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_cryptos(text: str) -> List[str]:
    text_norm = normalize_text(text)
    found: List[str] = []
    for crypto, keywords in CRYPTO_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw.lower())}\b", text_norm):
                found.append(crypto)
                break
    return found


def extract_descriptions_from_raw_xml(raw_xml: str) -> Dict[int, str]:
    """
    Extract <description> tags from raw RSS XML before feedparser normalizes them.
    Returns a dict mapping item index to description text.
    """
    descriptions: Dict[int, str] = {}
    try:
        root = ET.fromstring(raw_xml)
        # Handle both namespace and non-namespace versions
        items = root.findall(".//item")
        for idx, item in enumerate(items):
            desc_elem = item.find("description")
            if desc_elem is not None and desc_elem.text:
                descriptions[idx] = desc_elem.text
    except Exception:
        pass
    return descriptions


def fetch_feed_items() -> List[Tuple[str, dict, str]]:
    """
    Fetch RSS items. Returns list of (source, entry, original_description) tuples.
    original_description is from raw XML before feedparser normalization.
    """
    items: List[Tuple[str, dict, str]] = []
    items_per_feed = max(MAX_ITEMS_PER_FEED * 3, 200)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    for source, urls in RSS_FEEDS.items():
        print(f"Fetching {source} feeds ({len(urls)} URLs)...")
        for url in urls:
            parsed = None
            descriptions_map = {}
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                raw_text = response.text
                # Extract descriptions before feedparser normalizes them
                descriptions_map = extract_descriptions_from_raw_xml(raw_text)
                parsed = feedparser.parse(response.content)
            except Exception as request_error:
                # Fallback to direct feedparser URL parsing
                parsed = feedparser.parse(url)
                if getattr(parsed, "bozo", False):
                    print(f"  [skip] {url} -> request+parse failed: {request_error}")
                    continue

            if not parsed.entries:
                print(f"  [empty] {url} -> no entries")
                continue

            print(f"  [ok] {url} -> {len(parsed.entries)} entries")
            for idx, entry in enumerate(parsed.entries[:items_per_feed]):
                # Include original description from raw XML
                original_desc = descriptions_map.get(idx, "")
                items.append((source, entry, original_desc))
            # Light delay to reduce throttling from providers.
            time.sleep(0.2)
    return items


def extract_published_date(entry: dict) -> str:
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


def _extract_summary_raw(entry: dict, original_description: str = "") -> str:
    # Use original description from raw XML if available, before feedparser normalization
    summary = original_description

    if not summary:
        # Fallback: check feedparser-normalized fields
        summary = (
            entry.get("summary")
            or entry.get("description")
            or (entry.get("summary_detail") or {}).get("value")
            or entry.get("subtitle")
            or ""
        )

    if not summary and "content" in entry and entry["content"]:
        for part in entry["content"]:
            if isinstance(part, dict):
                part_value = part.get("value")
                if part_value:
                    summary = part_value
                    break

    return clean_html(summary)


def build_article_fields(entry: dict, original_description: str = "") -> Tuple[str, str, str]:
    title = clean_html(entry.get("title", "") or "")
    summary = _extract_summary_raw(entry, original_description)
    content = ""
    if "content" in entry and entry["content"]:
        content = " ".join(part.get("value", "") for part in entry["content"])
        content = clean_html(content)
    return title, summary, content


def collect_rss_articles(prefetched_items: List[Tuple[str, dict]] | None = None) -> List[RSSArticleRecord]:
    records: List[RSSArticleRecord] = []
    cutoff_date = (datetime.utcnow() - timedelta(days=MAX_DAYS_OLD)).date()
    seen = set()

    items = prefetched_items if prefetched_items is not None else fetch_feed_items()
    for source, entry, original_description in items:
        published_date = extract_published_date(entry)
        try:
            parsed_date = datetime.strptime(published_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        if parsed_date < cutoff_date:
            continue

        title, summary, content = build_article_fields(entry, original_description)
        url = entry.get("link", "") or ""
        if not title and not summary and not content:
            continue

        text_for_sentiment = f"{title}. {summary}. {content}".strip()
        if not text_for_sentiment:
            continue

        cryptos = detect_cryptos(text_for_sentiment)
        if not cryptos:
            continue

        for crypto in cryptos:
            dedupe_key = (url, crypto, published_date)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append(
                RSSArticleRecord(
                    source=source,
                    title=title,
                    summary=summary,
                    url=url,
                    published_date=published_date,
                    crypto=crypto,
                    text_for_sentiment=text_for_sentiment,
                )
            )

    return records


def collect_raw_rss_records(prefetched_items: List[Tuple[str, dict]] | None = None) -> List[RawRSSRecord]:
    records: List[RawRSSRecord] = []
    fetched_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    items = prefetched_items if prefetched_items is not None else fetch_feed_items()
    for source, entry, original_description in items:
        records.append(
            RawRSSRecord(
                source=source.lower(),
                url=entry.get("link", "") or "",
                published_at=extract_published_date(entry),
                fetched_at=fetched_at,
                title_raw=entry.get("title", "") or "",
                summary_raw=_extract_summary_raw(entry, original_description),
                raw_payload={"rss_entry": dict(entry)},
            )
        )
    return records


def save_to_csv(records: List[RSSArticleRecord], path: str = "rss_articles.csv") -> None:
    fieldnames = [
        "source",
        "title",
        "summary",
        "url",
        "published_date",
        "crypto",
        "text_for_sentiment",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "source": r.source,
                    "title": r.title,
                    "summary": r.summary,
                    "url": r.url,
                    "published_date": r.published_date,
                    "crypto": r.crypto,
                    "text_for_sentiment": r.text_for_sentiment,
                }
            )


def save_jsonl(records: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    records = collect_rss_articles()
    save_to_csv(records)
    print(f"Saved {len(records)} RSS records to rss_articles.csv")


if __name__ == "__main__":
    main()
