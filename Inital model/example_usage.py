"""
Example usage of the Sentiment Analysis API

This script demonstrates how to use the sentiment analysis API to analyze text sentiment.
Make sure the API server is running before executing this script.

Usage:
    python example_usage.py
"""

import requests
import json
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class SentimentAnalyzer:
    """Client for interacting with the Sentiment Analysis API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        """
        Initialize the sentiment analyzer client
        
        Args:
            base_url: Base URL of the sentiment analysis API
        """
        self.base_url = base_url.rstrip('/')
        self.analyze_url = f"{self.base_url}/analyze"
        self.health_url = f"{self.base_url}/health"
        self.models_url = f"{self.base_url}/models"
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health status of the API
        
        Returns:
            Dictionary containing health status information
        """
        try:
            response = requests.get(self.health_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to API at {self.base_url}. Is the server running?")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Health check failed: {str(e)}")
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available Ollama models
        
        Returns:
            Dictionary containing list of available models
        """
        try:
            response = requests.get(self.models_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to list models: {str(e)}")
    
    def analyze_sentiment(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of given text
        
        Args:
            text: Text to analyze
            model: Optional Ollama model to use (uses default if not provided)
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        payload = {
            "text": text
        }
        
        if model:
            payload["model"] = model
        
        try:
            response = requests.post(
                self.analyze_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for LLM processing
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            raise Exception(f"API error: {error_detail}")
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. The model may be too slow.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to API at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")


def print_sentiment_result(result: Dict[str, Any], text: str):
    """
    Pretty print sentiment analysis result
    
    Args:
        result: Result dictionary from API
        text: Original text that was analyzed
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS RESULT")
    print("="*60)
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Sentiment Label: {result['sentiment_label'].upper()}")
    print(f"Sentiment Score: {result['sentiment_score']:.3f}")
    print(f"Model Used: {result['model_used']}")
    if result.get('confidence'):
        print(f"Confidence: {result['confidence']:.3f}")
    print("="*60 + "\n")


def example_basic_usage():
    """Example: Basic sentiment analysis"""
    print("\n[Example 1] Basic Sentiment Analysis")
    print("-" * 60)
    
    analyzer = SentimentAnalyzer()
    
    # Check health first
    try:
        health = analyzer.check_health()
        print(f"API Status: {health.get('status')}")
        print(f"Ollama Status: {health.get('ollama')}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Analyze a positive text
    positive_text = "Bitcoin reached new all-time highs today! The market is extremely bullish and investors are excited about the future of cryptocurrency."
    try:
        result = analyzer.analyze_sentiment(positive_text)
        print_sentiment_result(result, positive_text)
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")


def example_negative_sentiment():
    """Example: Analyzing negative sentiment"""
    print("\n[Example 2] Negative Sentiment Analysis")
    print("-" * 60)
    
    analyzer = SentimentAnalyzer()
    
    negative_text = "The cryptocurrency market crashed today with Bitcoin dropping 15%. Investors are panicking and concerns about regulation are growing."
    try:
        result = analyzer.analyze_sentiment(negative_text)
        print_sentiment_result(result, negative_text)
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")


def example_neutral_sentiment():
    """Example: Analyzing neutral sentiment"""
    print("\n[Example 3] Neutral Sentiment Analysis")
    print("-" * 60)
    
    analyzer = SentimentAnalyzer()
    
    neutral_text = "Ethereum blockchain processed 1.2 million transactions today. The network remains stable with average gas prices at 25 gwei."
    try:
        result = analyzer.analyze_sentiment(neutral_text)
        print_sentiment_result(result, neutral_text)
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")


def example_custom_model():
    """Example: Using a custom model"""
    print("\n[Example 4] Using Custom Model")
    print("-" * 60)
    
    analyzer = SentimentAnalyzer()
    
    # First, list available models
    try:
        models_info = analyzer.list_models()
        available_models = models_info.get('models', [])
        print(f"Available models: {', '.join(available_models)}")
        
        if available_models:
            # Use the first available model (or specify a model name)
            model_to_use = available_models[0]
            print(f"Using model: {model_to_use}")
            
            text = "This is a test of sentiment analysis with a custom model."
            result = analyzer.analyze_sentiment(text, model=model_to_use)
            print_sentiment_result(result, text)
        else:
            print("No models available")
    except Exception as e:
        print(f"Error: {e}")


def example_batch_analysis():
    """Example: Analyzing multiple texts in batch"""
    print("\n[Example 5] Batch Sentiment Analysis")
    print("-" * 60)
    
    analyzer = SentimentAnalyzer()
    
    texts = [
        "Fantastic news! Our cryptocurrency portfolio is performing exceptionally well.",
        "Market conditions are uncertain and volatility is high.",
        "The new blockchain technology offers interesting opportunities for developers.",
        "Major exchange hack leads to millions in losses for traders.",
        "Daily trading volume increased by 5% compared to yesterday."
    ]
    
    results = []
    for i, text in enumerate(texts, 1):
        try:
            result = analyzer.analyze_sentiment(text)
            results.append(result)
            print(f"\n[{i}/{len(texts)}] {result['sentiment_label'].upper()} ({result['sentiment_score']:.3f})")
            print(f"Text: {text[:80]}...")
        except Exception as e:
            print(f"[{i}/{len(texts)}] Error: {e}")
    
    # Summary
    if results:
        avg_score = sum(r['sentiment_score'] for r in results) / len(results)
        positive_count = sum(1 for r in results if r['sentiment_label'] == 'positive')
        negative_count = sum(1 for r in results if r['sentiment_label'] == 'negative')
        neutral_count = sum(1 for r in results if r['sentiment_label'] == 'neutral')
        
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total texts analyzed: {len(results)}")
        print(f"Average sentiment score: {avg_score:.3f}")
        print(f"Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}")
        print("="*60)


def example_error_handling():
    """Example: Error handling"""
    print("\n[Example 6] Error Handling")
    print("-" * 60)
    
    analyzer = SentimentAnalyzer()
    
    # Test with empty text (should fail validation)
    try:
        result = analyzer.analyze_sentiment("")
        print_sentiment_result(result, "")
    except Exception as e:
        print(f"Expected error for empty text: {e}")
    
    # Test with invalid API URL
    invalid_analyzer = SentimentAnalyzer(base_url="http://localhost:9999")
    try:
        invalid_analyzer.check_health()
    except Exception as e:
        print(f"Expected error for invalid URL: {e}")


def main():
    """Main function to run all examples"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS API - USAGE EXAMPLES")
    print("="*60)
    print(f"API URL: {API_BASE_URL}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print("="*60)
    
    # Run examples
    example_basic_usage()
    example_negative_sentiment()
    example_neutral_sentiment()
    example_custom_model()
    example_batch_analysis()
    example_error_handling()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
