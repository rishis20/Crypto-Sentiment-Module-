CRYPTO_KEYWORDS = {
    "Bitcoin": ["bitcoin", "btc"],
    "Ethereum": ["ethereum", "eth"],
    "Solana": ["solana", "sol"],
}

# Reduced feed set for cleaning experiments.
PILOT_RSS_FEEDS = {
    "Reddit": [
        "https://www.reddit.com/r/CryptoCurrency/.rss",
    ],
    "CoinDesk": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ],
    "CoinTelegraph": [
        "https://cointelegraph.com/rss",
    ],
}

MAX_ITEMS_PER_FEED = 80
MAX_DAYS_OLD = 7
