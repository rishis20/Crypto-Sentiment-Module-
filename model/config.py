CRYPTO_KEYWORDS = {
    "Bitcoin": ["bitcoin", "btc", "BTC", "Bitcoin", "BITCOIN"],
    "Ethereum": ["ethereum", "eth", "ETH", "Ethereum", "ETHEREUM"],
    "Dogecoin": ["dogecoin", "doge", "DOGE", "Dogecoin", "DOGECOIN", "Doge"],
    "TetherUSDt": ["tetherusdt", "usdt", "tether", "USDT", "TetherUSDt", "TETHERUSDt"],
    "Dai": ["dai", "DAI","Dai"],
    "Solana": ["solana", "sol", "SOL", "Solana", "SOLANA"],
    "USDC": ["usdc", "USDC", "USDCOIN", "USDCoin"],
    "BNB": ["bnb", "binance coin", "binance", "BNB", "Binance Coin", "BINANCE COIN"],
    "XRP": ["xrp", "ripple", "XRP","Ripple","XRP Ledger"],
}

# Basic RSS feeds for crypto news and discussion
RSS_FEEDS = {
    "Reddit": [
        # General crypto subreddit
        "https://www.reddit.com/r/CryptoCurrency/.rss",
        "https://www.reddit.com/r/CryptoMarkets/.rss",
        # "https://www.reddit.com/r/crypto/.rss",
        # "https://www.reddit.com/r/Crypto_com/.rss",
        # "https://www.reddit.com/r/USDT_EXCHANGE/.rss",
        # "https://www.reddit.com/r/binance/.rss",
        # "https://www.reddit.com/r/BinanceUS/.rss",
        # "https://www.reddit.com/r/BinanceCrypto/.rss",
        # "https://www.reddit.com/r/StableCoins/.rss",
        # "https://www.reddit.com/r/defi/.rss",
        # "https://www.reddit.com/r/solana/.rss",
        # "https://www.reddit.com/r/SolanaNFT/.rss",
        # "https://www.reddit.com/r/bnbchainofficial/.rss",
        # "https://www.reddit.com/r/BNBinance/.rss",
        # "https://www.reddit.com/r/XRP/.rss",
        # "https://www.reddit.com/r/XRPUnite/.rss",
        # "https://www.reddit.com/r/Bitcoin/.rss",
        # "https://www.reddit.com/r/ethereum/.rss",
        # "https://www.reddit.com/r/dogecoin/.rss",
        # "https://www.reddit.com/r/dogecoindev/.rss",
    ],
    "CoinDesk": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://www.coindesk.com/markets/rss",
        # "https://www.coindesk.com/tech/rss",
        # "https://www.coindesk.com/policy/rss",
        # "https://www.coindesk.com/business/rss",
        # "https://www.coindesk.com/learn/rss",
    ],
    "CoinTelegraph": [
        "https://cointelegraph.com/rss",
        "https://cointelegraph.com/rss/tag/bitcoin",
    #     "https://cointelegraph.com/rss/tag/ethereum",
    #     "https://cointelegraph.com/rss/tag/blockchain",
    #     "https://cointelegraph.com/rss/tag/altcoin",
    #     "https://cointelegraph.com/rss/tag/regulation",
    #     "https://cointelegraph.com/rss/category/analysis",
    #     "https://cointelegraph.com/rss/category/market-analysis",
    #     "https://cointelegraph.com/rss/category/price-analysis",
    #     "https://cointelegraph.com/rss/category/features",
    ],
}

# Number of items to process per feed (to keep runtime reasonable)
MAX_ITEMS_PER_FEED = 50

# Test push