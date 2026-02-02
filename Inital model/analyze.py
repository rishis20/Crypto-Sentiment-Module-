"""
Sentiment Analysis API using Ollama
Generates sentiment scores for given text using Ollama LLM models
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
import httpx
import json
import re
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using Ollama LLM models",
    version="1.0.0"
)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # Default model, can be changed


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1)
    model: Optional[str] = Field(None, description="Ollama model to use (optional)")


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    text: str
    sentiment_score: float = Field(..., description="Sentiment score from -1.0 (negative) to 1.0 (positive)")
    sentiment_label: str = Field(..., description="Sentiment label: 'positive', 'negative', or 'neutral'")
    model_used: str
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0)")


def normalize_score(score: float) -> float:
    """Normalize score to -1.0 to 1.0 range"""
    if abs(score) > 1.0:
        score = score / 10.0 if abs(score) <= 10.0 else score / 100.0
    return max(-1.0, min(1.0, score))


def extract_score_from_json(response_text: str) -> Optional[float]:
    """Try to extract score from JSON format"""
    try:
        json_match = re.search(r'\{[^}]*"score"\s*:\s*([-+]?\d*\.?\d+)[^}]*\}', response_text, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            score = float(data.get("score", 0.0))
            return normalize_score(score)
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def extract_score_from_patterns(response_text: str) -> Optional[float]:
    """Try to extract score using regex patterns"""
    score_patterns = [
        r'"score"\s*:\s*([-+]?\d+\.?\d*)',
        r'score["\']?\s*:\s*([-+]?\d+\.?\d*)',
        r'score["\']?\s*[=]\s*([-+]?\d+\.?\d*)',
    ]
    
    for pattern in score_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            try:
                score = float(matches[0])
                return normalize_score(score)
            except (ValueError, OverflowError, TypeError):
                continue
    return None


def extract_score_from_decimals(response_text: str) -> Optional[float]:
    """Try to find standalone decimal numbers that could be scores"""
    decimal_pattern = r'[-+]?\d+\.\d+'
    matches = re.findall(decimal_pattern, response_text)
    for match in matches:
        try:
            score = float(match)
            if -1.0 <= score <= 1.0:
                return score
            if 0 <= score <= 10:
                return normalize_score((score - 5.0) * 2.0)
        except ValueError:
            continue
    return None


def extract_sentiment_from_keywords(response_text: str) -> tuple[float, Optional[float]]:
    """Fallback: analyze text content for sentiment keywords"""
    response_lower = response_text.lower()
    
    if any(word in response_lower for word in ['very positive', 'extremely positive', 'strongly positive', 'highly positive']):
        return 0.8, 0.5
    if any(word in response_lower for word in ['positive', 'good', 'bullish', 'upward', 'gains', 'rises', 'wins']):
        return 0.5, 0.4
    if any(word in response_lower for word in ['very negative', 'extremely negative', 'strongly negative', 'highly negative']):
        return -0.8, 0.5
    if any(word in response_lower for word in ['negative', 'bad', 'bearish', 'downward', 'losses', 'drops', 'fails', 'crashes']):
        return -0.5, 0.4
    if any(word in response_lower for word in ['neutral', 'mixed', 'uncertain', 'balanced']):
        return 0.0, 0.5
    
    return 0.0, 0.3


def extract_sentiment_score(response_text: str) -> tuple[float, Optional[float]]:
    """
    Extract sentiment score from Ollama response
    
    Args:
        response_text: Raw response from Ollama
        
    Returns:
        Tuple of (sentiment_score, confidence) where score is -1.0 to 1.0
    """
    if not response_text:
        return 0.0, None
    
    # Try JSON extraction first
    score = extract_score_from_json(response_text)
    if score is not None:
        return score, None
    
    # Try pattern-based extraction
    score = extract_score_from_patterns(response_text)
    if score is not None:
        return score, None
    
    # Try decimal extraction
    score = extract_score_from_decimals(response_text)
    if score is not None:
        return score, None
    
    # Fallback to keyword-based sentiment
    return extract_sentiment_from_keywords(response_text)


def get_sentiment_label(score: float) -> str:
    """
    Convert sentiment score to label
    
    Args:
        score: Sentiment score from -1.0 to 1.0
        
    Returns:
        Sentiment label: 'positive', 'negative', or 'neutral'
    """
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    else:
        return "neutral"


def create_sentiment_prompt(text: str) -> str:
    """
    Create a detailed and accurate sentiment analysis prompt
    
    Args:
        text: Text to analyze for sentiment
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a sentiment analysis expert specializing in financial and cryptocurrency markets. Analyze the sentiment of the following text and provide a precise sentiment score.

Text to analyze: "{text}"

Guidelines for sentiment scoring:
- Score range: -1.0 to 1.0 (decimal number)
- Positive sentiment (0.1 to 1.0): Optimistic, bullish, positive news, gains, upward trends, buying opportunities, success stories, regulatory approvals, adoption, partnerships
- Negative sentiment (-1.0 to -0.1): Pessimistic, bearish, negative news, losses, crashes, downward trends, scams, frauds, regulatory bans, security breaches, failures
- Neutral sentiment (around 0.0): Factual statements, questions, neutral information, price movements without emotional context, technical descriptions

Scoring scale examples:
- Very positive (0.7 to 1.0): Strong bullish sentiment, major gains, breakthrough news
- Moderately positive (0.3 to 0.7): Positive outlook, expected gains, good news
- Slightly positive (0.1 to 0.3): Mildly optimistic, minor positive indicators
- Neutral (-0.1 to 0.1): Factual, informational, balanced, uncertain
- Slightly negative (-0.3 to -0.1): Mild concerns, minor setbacks
- Moderately negative (-0.7 to -0.3): Negative outlook, expected losses, bad news
- Very negative (-1.0 to -0.7): Strong bearish sentiment, major losses, severe negative events

Important considerations:
- For financial/crypto context: Price increases, adoption, partnerships, regulatory approval = positive
- Price decreases, crashes, hacks, regulatory bans, scams = negative
- Questions seeking advice are typically neutral unless they express concern or excitement
- Headlines with positive action words (wins, gains, rises, approves) = positive
- Headlines with negative action words (fails, drops, crashes, loses) = negative
- Factual statements about prices or events without emotional tone = neutral (0.0 to ±0.2)

You MUST respond with ONLY a valid JSON object in this exact format (no additional text, explanations, or formatting):
{{
    "score": <your_score_as_decimal_number>
}}

Provide the score now:"""
    
    return prompt


async def analyze_text_sentiment(text: str, model_name: Optional[str] = None) -> float:
    """
    Analyze sentiment of a single text string
    
    Args:
        text: Text to analyze
        model_name: Optional model name to use
        
    Returns:
        Sentiment score from -1.0 to 1.0
    """
    model_to_use = model_name or OLLAMA_MODEL
    
    # Skip empty text
    if not text or pd.isna(text) or str(text).strip() == "":
        return 0.0
    
    # Prepare prompt for sentiment analysis using improved prompt
    prompt = create_sentiment_prompt(str(text))

    try:
        # Call Ollama API
        ollama_url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Very low temperature for consistent, deterministic results
                "top_p": 0.95,  # High top_p for focused responses
                "top_k": 40,  # Limit vocabulary for more consistent outputs
                "repeat_penalty": 1.1  # Slight penalty to avoid repetition
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ollama_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            model_response = result.get("response", "")
        
        # Extract sentiment score from response
        sentiment_score, _ = extract_sentiment_score(model_response)
        return sentiment_score
        
    except Exception as e:
        print(f"Error analyzing sentiment for text: {str(e)}")
        return 0.0


async def process_csv_file(csv_file_path: str, output_path: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """
    Process CSV file: analyze first column for sentiment and write scores to last column
    
    Args:
        csv_file_path: Path to input CSV file
        output_path: Optional path for output CSV (defaults to input path with _analyzed suffix)
        model_name: Optional model name to use
        
    Returns:
        Path to output CSV file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Get first column name
        first_column = df.columns[0]
        
        # Always add a new column at the end for sentiment scores (making it the last column)
        # This avoids overwriting existing data
        output_column = "sentiment_score"
        
        print(f"Processing CSV: {csv_file_path}")
        print(f"Using first column: {first_column}")
        print(f"Writing to column: {output_column}")
        print(f"Total rows: {len(df)}")
        
        # Process each row
        sentiment_scores = []
        for idx, row in df.iterrows():
            text_value = row[first_column]
            print(f"Processing row {idx + 1}/{len(df)}: {str(text_value)[:50]}...")
            
            # Analyze sentiment
            score = await analyze_text_sentiment(text_value, model_name)
            sentiment_scores.append(score)
        
        # Add or update sentiment scores column
        df[output_column] = sentiment_scores
        
        # Determine output path
        if output_path is None:
            input_path = Path(csv_file_path)
            output_path = str(input_path.parent / f"{input_path.stem}_analyzed{input_path.suffix}")
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved analyzed CSV to: {output_path}")
        
        return output_path
        
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Ollama is accessible
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "connected" if response.status_code == 200 else "disconnected"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "ollama": ollama_status,
        "ollama_url": OLLAMA_BASE_URL
    }


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of given text using Ollama
    
    Args:
        request: SentimentRequest containing text and optional model name
        
    Returns:
        SentimentResponse with sentiment score and label
    """
    model_name = request.model or OLLAMA_MODEL
    
    # Prepare prompt for sentiment analysis using improved prompt
    prompt = create_sentiment_prompt(request.text)

    try:
        # Call Ollama API
        ollama_url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Very low temperature for consistent, deterministic results
                "top_p": 0.95,  # High top_p for focused responses
                "top_k": 40,  # Limit vocabulary for more consistent outputs
                "repeat_penalty": 1.1  # Slight penalty to avoid repetition
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ollama_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            model_response = result.get("response", "")
        
        # Extract sentiment score from response
        sentiment_score, confidence = extract_sentiment_score(model_response)
        sentiment_label = get_sentiment_label(sentiment_score)
        
        return SentimentResponse(
            text=request.text,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            model_used=model_name,
            confidence=confidence
        )
        
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama server at {OLLAMA_BASE_URL}. Please ensure Ollama is running."
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request to Ollama timed out. The model may be too slow or unavailable."
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing sentiment analysis: {str(e)}"
        )


@app.get("/models")
async def list_available_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [model.get("name", "") for model in data.get("models", [])]
            return {
                "models": models,
                "default_model": OLLAMA_MODEL
            }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama server: {str(e)}"
        )


class CSVProcessRequest(BaseModel):
    """Request model for CSV processing"""
    csv_file_path: str = Field(..., description="Path to input CSV file")
    output_path: Optional[str] = Field(None, description="Optional path for output CSV (defaults to input_path with _analyzed suffix)")
    model: Optional[str] = Field(None, description="Ollama model to use (optional)")


class CSVProcessResponse(BaseModel):
    """Response model for CSV processing"""
    message: str
    input_file: str
    output_file: str
    rows_processed: int
    model_used: str


@app.post("/process_csv", response_model=CSVProcessResponse)
async def process_csv_endpoint(request: CSVProcessRequest):
    """
    Process CSV file: analyze first column for sentiment and write scores to last column
    
    Args:
        request: CSVProcessRequest containing CSV file path and optional parameters
        
    Returns:
        CSVProcessResponse with processing results
    """
    model_name = request.model or OLLAMA_MODEL
    
    try:
        # Check if input file exists
        if not os.path.exists(request.csv_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"CSV file not found: {request.csv_file_path}"
            )
        
        # Process CSV file
        output_path = await process_csv_file(
            request.csv_file_path,
            request.output_path,
            model_name
        )
        
        # Count rows processed
        df = pd.read_csv(request.csv_file_path)
        rows_processed = len(df)
        
        return CSVProcessResponse(
            message="CSV file processed successfully",
            input_file=request.csv_file_path,
            output_file=output_path,
            rows_processed=rows_processed,
            model_used=model_name
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    import sys
    import asyncio
    
    # Check if CSV file path is provided as command line argument
    if len(sys.argv) > 1:
        # Command-line mode: process CSV file directly
        csv_file_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        model_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        async def main():
            try:
                result = await process_csv_file(csv_file_path, output_path, model_name)
                print("\n✓ Successfully processed CSV file!")
                print(f"  Input: {csv_file_path}")
                print(f"  Output: {result}")
            except Exception as e:
                print(f"\n✗ Error processing CSV file: {str(e)}", file=sys.stderr)
                sys.exit(1)
        
        asyncio.run(main())
    else:
        # API server mode
        port = int(os.getenv("SENTIMENT_API_PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)