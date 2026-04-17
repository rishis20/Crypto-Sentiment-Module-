"""
Small RSS cleaning experiment pipeline.

Writes separate artifacts:
- data/raw/news_raw_YYYY-MM-DD.jsonl
- data/clean/news_clean_YYYY-MM-DD.jsonl
"""

from datetime import datetime, timedelta
import hashlib
import json
import os
import re
import time
from urllib.parse import urlparse, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup

from pilot_config import CRYPTO_KEYWORDS, PILOT_RSS_FEEDS, MAX_ITEMS_PER_FEED, MAX_DAYS_OLD

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


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ", strip=True)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    # Keep scheme + host + path, drop query/fragment to reduce duplicate noise.
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


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


def detect_cryptos(text: str) -> list[str]:
    text_norm = normalize_text(text)
    found = []
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
    # If summary has almost no alphabetic text after cleanup, it's likely metadata-only.
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
    source: str, title_clean: str, summary_clean: str, text_for_model: str, cryptos: list[str]
) -> tuple[bool, str]:
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


def fetch_feed_items() -> list[tuple[str, str, dict]]:
    items: list[tuple[str, str, dict]] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    for source, urls in PILOT_RSS_FEEDS.items():
        print(f"Fetching {source} feeds ({len(urls)} URLs)...")
        for url in urls:
            parsed = None
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                parsed = feedparser.parse(response.content)
            except Exception as request_error:
                parsed = feedparser.parse(url)
                if getattr(parsed, "bozo", False):
                    print(f"  [skip] {url} -> request+parse failed: {request_error}")
                    continue

            if not parsed.entries:
                print(f"  [empty] {url} -> no entries")
                continue

            print(f"  [ok] {url} -> {len(parsed.entries)} entries")
            for entry in parsed.entries[:MAX_ITEMS_PER_FEED]:
                items.append((source, url, entry))
            time.sleep(0.2)

    return items


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run() -> None:
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "data", "raw")
    clean_dir = os.path.join(base_dir, "data", "clean")
    reports_dir = os.path.join(base_dir, "data", "reports")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    fetched_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    raw_path = os.path.join(raw_dir, f"news_raw_{date_tag}.jsonl")
    clean_path = os.path.join(clean_dir, f"news_clean_{date_tag}.jsonl")
    reject_path = os.path.join(clean_dir, f"news_rejected_{date_tag}.jsonl")
    stats_path = os.path.join(reports_dir, f"cleaning_stats_{date_tag}.json")

    items = fetch_feed_items()
    raw_rows: list[dict] = []
    clean_rows: list[dict] = []
    rejected_rows: list[dict] = []

    cutoff_date = (datetime.utcnow() - timedelta(days=MAX_DAYS_OLD)).date()
    seen_hashes: set[str] = set()
    # Cross-run dedupe: include hashes from previous clean artifacts.
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
                    h = row.get("dedupe_hash")
                    if h:
                        seen_hashes.add(h)
    except FileNotFoundError:
        pass

    source_raw_counts: dict[str, int] = {}
    source_clean_counts: dict[str, int] = {}
    reject_reason_counts: dict[str, int] = {}

    for source, feed_url, entry in items:
        source_key = source.lower()
        source_raw_counts[source_key] = source_raw_counts.get(source_key, 0) + 1
        published_at = extract_published_date(entry)
        url = entry.get("link", "") or ""
        canonical_url = canonicalize_url(url)
        title_raw = entry.get("title", "") or ""
        summary_raw = (entry.get("summary", "") or entry.get("description", "") or "")

        raw_rows.append(
            {
                "source": source.lower(),
                "feed_url": feed_url,
                "url": canonical_url or url,
                "published_at": published_at,
                "fetched_at": fetched_at,
                "title_raw": title_raw,
                "summary_raw": summary_raw,
                "raw_payload": {"rss_entry": dict(entry)},
            }
        )

        # Light cleaning/validation (no advanced preprocessing yet).
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

        title_clean = clean_html(title_raw)
        summary_clean = clean_html(summary_raw)
        title_clean = remove_boilerplate(title_clean)
        summary_clean = remove_boilerplate(summary_clean)
        text_for_model = f"{title_clean}\n\n{summary_clean}".strip()

        cryptos = detect_cryptos(text_for_model)
        is_valid, reason = validate_clean_candidate(
            source.lower(), title_clean, summary_clean, text_for_model, cryptos
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
                "fetched_at": fetched_at,
                "title_clean": title_clean,
                "summary_clean": summary_clean,
                "text_for_model": text_for_model,
                "cryptos": cryptos,
                "dedupe_hash": dedupe_hash,
            }
        )

    write_jsonl(raw_path, raw_rows)
    write_jsonl(clean_path, clean_rows)
    write_jsonl(reject_path, rejected_rows)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": date_tag,
                "fetched_at": fetched_at,
                "totals": {
                    "raw_rows": len(raw_rows),
                    "clean_rows": len(clean_rows),
                    "rejected_rows": len(rejected_rows),
                },
                "by_source": {
                    "raw": source_raw_counts,
                    "clean": source_clean_counts,
                },
                "reject_reason_counts": reject_reason_counts,
                "min_char_count": MIN_CHAR_COUNT,
                "min_alpha_ratio": MIN_ALPHA_RATIO,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved raw JSONL: {len(raw_rows)} -> {raw_path}")
    print(f"Saved clean JSONL: {len(clean_rows)} -> {clean_path}")
    print(f"Saved rejected JSONL: {len(rejected_rows)} -> {reject_path}")
    print(f"Saved stats JSON: {stats_path}")


if __name__ == "__main__":
    run()
