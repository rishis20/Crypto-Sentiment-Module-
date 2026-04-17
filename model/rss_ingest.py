"""
RSS ingestion for Ollama-only sentiment pipeline.

This module collects crypto-related RSS entries and exports a CSV that can be
scored by analyze.py. It does not run any sentiment model itself.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import csv
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup

from config import CRYPTO_KEYWORDS, RSS_FEEDS, MAX_ITEMS_PER_FEED

MAX_DAYS_OLD = 7
MIN_CHAR_COUNT = 80
MIN_ALPHA_RATIO = 0.60

REDDIT_THREAD_PATTERNS = [
    r"\bdaily crypto discussion\b",
    r"\bweekly\b.*\bdiscussion\b",
    r"\bmonthly\b.*\bdiscussion\b",
]

BOILERPLATE_PATTERNS = [
    r"\bsubmitted by\s+/u/\w+",
    r"\[link\]",
    r"\[comments\]",
    r"\bread more\b",
    r"\bjoin (our )?discord\b",
]

IMAGE_ONLY_PATTERNS = [
    r"^so real\.*$",
]

ENGLISH_HINT_WORDS = {
    "the",
    "and",
    "is",
    "are",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "from",
}


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
    feed_url: str
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


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def detect_cryptos(text: str) -> List[str]:
    text_norm = normalize_text(text)
    found: List[str] = []
    for crypto, keywords in CRYPTO_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw.lower())}\b", text_norm):
                found.append(crypto)
                break
    return found


def remove_boilerplate(text: str) -> str:
    cleaned = text or ""
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_discussion_thread(title: str) -> bool:
    title_norm = normalize_text(title)
    return any(re.search(p, title_norm) for p in REDDIT_THREAD_PATTERNS)


def looks_image_or_link_only(title: str, summary: str) -> bool:
    title_norm = normalize_text(title)
    summary_norm = normalize_text(summary)
    if any(re.search(p, title_norm) for p in IMAGE_ONLY_PATTERNS):
        return True
    alpha_chars = len(re.findall(r"[a-zA-Z]", summary_norm))
    return alpha_chars < 20


def stable_dedupe_hash(title_clean: str, summary_clean: str) -> str:
    basis = normalize_text(f"{title_clean} {summary_clean}")
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def likely_english(text: str) -> bool:
    text = text or ""
    if not text.strip():
        return False
    alpha = len(re.findall(r"[A-Za-z]", text))
    total = len(text)
    if total == 0:
        return False
    alpha_ratio = alpha / total
    words = set(normalize_text(text).split())
    hint_hits = len(words.intersection(ENGLISH_HINT_WORDS))
    return alpha_ratio >= MIN_ALPHA_RATIO and hint_hits >= 2


def validate_clean_candidate(
    source: str, title_clean: str, summary_clean: str, text_for_model: str, cryptos: List[str]
) -> Tuple[bool, str]:
    if source == "reddit" and is_discussion_thread(title_clean):
        return False, "boilerplate_thread"
    if source == "reddit" and looks_image_or_link_only(title_clean, summary_clean):
        return False, "image_or_link_only"
    if len(text_for_model) < MIN_CHAR_COUNT:
        return False, "too_short"
    if not likely_english(text_for_model):
        return False, "non_english_or_garbled"
    if not cryptos:
        return False, "non_crypto_after_clean"
    return True, "accepted"


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


def fetch_feed_items() -> List[Tuple[str, str, dict, str]]:
    """
    Fetch RSS items. Returns list of (source, feed_url, entry, original_description) tuples.
    original_description is from raw XML before feedparser normalization.
    """
    items: List[Tuple[str, str, dict, str]] = []
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
                items.append((source, url, entry, original_desc))
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


def collect_rss_articles(prefetched_items: List[Tuple[str, str, dict, str]] | None = None) -> List[RSSArticleRecord]:
    records: List[RSSArticleRecord] = []
    cutoff_date = (datetime.utcnow() - timedelta(days=MAX_DAYS_OLD)).date()
    seen = set()

    items = prefetched_items if prefetched_items is not None else fetch_feed_items()
    for source, _feed_url, entry, original_description in items:
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


def collect_raw_rss_records(prefetched_items: List[Tuple[str, str, dict, str]] | None = None) -> List[RawRSSRecord]:
    records: List[RawRSSRecord] = []
    fetched_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    items = prefetched_items if prefetched_items is not None else fetch_feed_items()
    for source, feed_url, entry, original_description in items:
        records.append(
            RawRSSRecord(
                source=source.lower(),
                feed_url=feed_url,
                url=entry.get("link", "") or "",
                published_at=extract_published_date(entry),
                fetched_at=fetched_at,
                title_raw=entry.get("title", "") or "",
                summary_raw=_extract_summary_raw(entry, original_description),
                raw_payload={"rss_entry": dict(entry)},
            )
        )
    return records


def load_seen_hashes(clean_dir: str) -> set[str]:
    seen_hashes: set[str] = set()
    try:
        for name in os.listdir(clean_dir):
            if not name.startswith("news_clean_") or not name.endswith(".jsonl"):
                continue
            prior_path = os.path.join(clean_dir, name)
            with open(prior_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    dedupe_hash = row.get("dedupe_hash")
                    if dedupe_hash:
                        seen_hashes.add(dedupe_hash)
    except FileNotFoundError:
        pass
    return seen_hashes


def build_clean_dataset(
    prefetched_items: List[Tuple[str, str, dict, str]] | None = None,
    clean_dir: str | None = None,
    fetched_at: str | None = None,
) -> Tuple[List[dict], List[dict], Dict[str, Dict[str, int] | int | Dict[str, int]]]:
    items = prefetched_items if prefetched_items is not None else fetch_feed_items()
    cutoff_date = (datetime.utcnow() - timedelta(days=MAX_DAYS_OLD)).date()
    fetched_at_value = fetched_at or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    seen_hashes: set[str] = set()
    if clean_dir:
        seen_hashes = load_seen_hashes(clean_dir)

    clean_rows: List[dict] = []
    rejected_rows: List[dict] = []
    source_raw_counts: Dict[str, int] = {}
    source_clean_counts: Dict[str, int] = {}
    reject_reason_counts: Dict[str, int] = {}

    for source, _feed_url, entry, original_description in items:
        source_key = source.lower()
        source_raw_counts[source_key] = source_raw_counts.get(source_key, 0) + 1
        published_at = extract_published_date(entry)
        url = entry.get("link", "") or ""
        canonical_url = canonicalize_url(url)

        try:
            parsed_date = datetime.strptime(published_at, "%Y-%m-%d").date()
        except ValueError:
            rejected_rows.append(
                {
                    "source": source_key,
                    "url": canonical_url or url,
                    "published_at": published_at,
                    "reject_reason": "invalid_date",
                }
            )
            reject_reason_counts["invalid_date"] = reject_reason_counts.get("invalid_date", 0) + 1
            continue
        if parsed_date < cutoff_date:
            rejected_rows.append(
                {
                    "source": source_key,
                    "url": canonical_url or url,
                    "published_at": published_at,
                    "reject_reason": "outside_date_window",
                }
            )
            reject_reason_counts["outside_date_window"] = reject_reason_counts.get("outside_date_window", 0) + 1
            continue

        title_clean, summary_clean, _content_clean = build_article_fields(entry, original_description)
        title_clean = remove_boilerplate(title_clean)
        summary_clean = remove_boilerplate(summary_clean)
        text_for_model = f"{title_clean}\n\n{summary_clean}".strip()

        cryptos = detect_cryptos(text_for_model)
        is_valid, reason = validate_clean_candidate(
            source_key, title_clean, summary_clean, text_for_model, cryptos
        )
        if not is_valid:
            rejected_rows.append(
                {
                    "source": source_key,
                    "url": canonical_url or url,
                    "published_at": published_at,
                    "title_clean": title_clean,
                    "summary_clean": summary_clean,
                    "reject_reason": reason,
                }
            )
            reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1
            continue

        dedupe_hash = stable_dedupe_hash(title_clean, summary_clean)
        if dedupe_hash in seen_hashes:
            rejected_rows.append(
                {
                    "source": source_key,
                    "url": canonical_url or url,
                    "published_at": published_at,
                    "title_clean": title_clean,
                    "summary_clean": summary_clean,
                    "reject_reason": "duplicate",
                    "dedupe_hash": dedupe_hash,
                }
            )
            reject_reason_counts["duplicate"] = reject_reason_counts.get("duplicate", 0) + 1
            continue
        seen_hashes.add(dedupe_hash)

        source_clean_counts[source_key] = source_clean_counts.get(source_key, 0) + 1
        clean_rows.append(
            {
                "source": source_key,
                "url": canonical_url or url,
                "published_at": published_at,
                "fetched_at": fetched_at_value,
                "title_clean": title_clean,
                "summary_clean": summary_clean,
                "text_for_model": text_for_model,
                "cryptos": cryptos,
                "dedupe_hash": dedupe_hash,
            }
        )

    stats: Dict[str, Dict[str, int] | int | Dict[str, int]] = {
        "raw_rows": len(items),
        "clean_rows": len(clean_rows),
        "rejected_rows": len(rejected_rows),
        "by_source_raw": source_raw_counts,
        "by_source_clean": source_clean_counts,
        "reject_reason_counts": reject_reason_counts,
    }
    return clean_rows, rejected_rows, stats


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
