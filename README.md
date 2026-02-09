# Sentiment Analysis API using Ollama

A FastAPI-based sentiment analysis service that uses Ollama LLM models to generate sentiment scores from text. Perfect for analyzing financial news, social media posts, and other text content for trading and investment decisions.

## Features

- **LLM-powered sentiment analysis** using Ollama models
- **Sentiment scores** ranging from -1.0 (very negative) to 1.0 (very positive)
- **Automatic labeling** (positive, negative, or neutral)
- **Async API** built with FastAPI for high performance
- **Configurable** model selection and Ollama server URL
- **Robust error handling** for connection issues and timeouts
- **Interactive API documentation** via FastAPI's Swagger UI

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running
   - Download from [https://ollama.ai](https://ollama.ai)
   - Ensure Ollama is running: `ollama serve`
3. **At least one Ollama model** downloaded
   - Default: `llama3.2` (or specify your preferred model)
   - Download a model: `ollama pull llama3.2`

## Project structure

- **Project root** (`Crypto-Sentiment-Module-/`): contains `README.md` and `requirements.txt`
- **`Inital model/`**: contains the API and scripts
  - `analyze.py` — FastAPI sentiment API (Ollama)
  - `example_usage.py` — example client for the API

## Installation

1. **Navigate to the project root** (the folder that contains `requirements.txt` and `Inital model`):
   ```bash
   cd /path/to/Crypto-Sentiment-Module-
   ```

2. **Install Python dependencies** (from project root):
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```
   If this returns a JSON response, Ollama is running correctly.

## Configuration

Create a `.env` file in the **`Inital model`** directory (optional), or in the project root if you run the server from there:

```env
# Ollama server URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Default Ollama model to use (default: llama3.2)
OLLAMA_MODEL=llama3.2

# API server port (default: 8000)
SENTIMENT_API_PORT=8000
```

If no `.env` file is provided, the defaults above will be used.

## Usage

### Starting the API Server

Run the server from the **`Inital model`** directory so `analyze.py` and `.env` are found:

```bash
cd "Inital model"
```

**Option 1: Using Python directly**
```bash
python analyze.py
```

**Option 2: Using uvicorn**
```bash
uvicorn analyze:app --host 0.0.0.0 --port 8000 --reload
```

The API will start on `http://localhost:8000` (or your configured port).

### Accessing API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Analyze Sentiment

**POST** `/analyze`

Analyzes the sentiment of given text and returns a sentiment score.

**Request Body:**
```json
{
  "text": "This is great news for the cryptocurrency market!",
  "model": "llama3.2"  // optional, uses default if not provided
}
```

**Response:**
```json
{
  "text": "This is great news for the cryptocurrency market!",
  "sentiment_score": 0.75,
  "sentiment_label": "positive",
  "model_used": "llama3.2",
  "confidence": 0.85
}
```

**Response Fields:**
- `text`: The original input text
- `sentiment_score`: Float from -1.0 (very negative) to 1.0 (very positive)
- `sentiment_label`: One of "positive", "negative", or "neutral"
- `model_used`: The Ollama model that was used
- `confidence`: Optional confidence score (0.0 to 1.0)

### 2. Health Check

**GET** `/health`

Checks the health status of the API and Ollama connection.

**Response:**
```json
{
  "status": "ok",
  "ollama": "connected",
  "ollama_url": "http://localhost:11434"
}
```

### 3. List Available Models

**GET** `/models`

Returns a list of available Ollama models.

**Response:**
```json
{
  "models": ["llama3.2", "mistral", "codellama"],
  "default_model": "llama3.2"
}
```

## Examples

### Using cURL

**Analyze sentiment:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitcoin price dropped 10% today, investors are worried."
  }'
```

**With custom model:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The new blockchain technology shows promising potential.",
    "model": "mistral"
  }'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**List models:**
```bash
curl http://localhost:8000/models
```

### Using Python

```python
import requests

# Analyze sentiment
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "Ethereum upgrade successfully deployed!",
        "model": "llama3.2"
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment_label']}")
print(f"Score: {result['sentiment_score']}")
```

### Using JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Market sentiment is bullish today!',
    model: 'llama3.2'
  })
});

const result = await response.json();
console.log(`Sentiment: ${result.sentiment_label}`);
console.log(`Score: ${result.sentiment_score}`);
```

## Sentiment Score Interpretation

| Score Range | Label | Interpretation |
|------------|-------|----------------|
| > 0.2 | Positive | Generally positive sentiment |
| -0.2 to 0.2 | Neutral | Mixed or neutral sentiment |
| < -0.2 | Negative | Generally negative sentiment |

## Error Handling

The API handles various error scenarios:

### Connection Error (503)
```json
{
  "detail": "Cannot connect to Ollama server at http://localhost:11434. Please ensure Ollama is running."
}
```
**Solution:** Make sure Ollama is running: `ollama serve`

### Timeout Error (504)
```json
{
  "detail": "Request to Ollama timed out. The model may be too slow or unavailable."
}
```
**Solution:** Try a smaller/faster model or increase timeout settings

### Model Not Found (404)
```json
{
  "detail": "Ollama API error: model not found"
}
```
**Solution:** Download the model first: `ollama pull <model-name>`

### Invalid Request (422)
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```
**Solution:** Ensure the request body contains a non-empty `text` field

## Integration with Express Backend

To integrate this API with your Express.js backend, you can call it from your Node.js routes:

```javascript
// In your Express route file
const axios = require('axios');

async function analyzeSentiment(text) {
  try {
    const response = await axios.post('http://localhost:8000/analyze', {
      text: text,
      model: 'llama3.2'
    });
    return response.data;
  } catch (error) {
    console.error('Sentiment analysis error:', error.message);
    throw error;
  }
}

// Use in a route
router.post('/api/sentiment/analyze', async (req, res) => {
  try {
    const { text } = req.body;
    const result = await analyzeSentiment(text);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Failed to analyze sentiment' });
  }
});
```

## Troubleshooting

### Ollama Connection Issues

1. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Check Ollama logs:**
   ```bash
   ollama serve
   ```

3. **Verify model is downloaded:**
   ```bash
   ollama list
   ```

4. **Download a model if needed:**
   ```bash
   ollama pull llama3.2
   ```

### API Server Issues

1. **Check if port is already in use:**
   ```bash
   lsof -i :8000
   ```

2. **View API logs** in the terminal where the server is running

3. **Check environment variables** are set correctly in `.env` (in `Inital model/` or project root)

## Performance Considerations

- **Model Selection**: Larger models (e.g., `llama3.2`) provide better accuracy but are slower. Smaller models are faster but may be less accurate.
- **Concurrent Requests**: FastAPI handles multiple requests concurrently using async/await.
- **Caching**: Consider implementing response caching for repeated analyses of the same text.
- **Timeout**: Default timeout is 30 seconds. Adjust based on your model's speed.

## Development

### Running in Development Mode

From the **`Inital model`** directory:

```bash
cd "Inital model"
uvicorn analyze:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload on code changes.

### Testing

Test the API endpoints using the interactive docs at http://localhost:8000/docs or use curl/Postman.

## License

This module is part of the Blocktrade project.

## Support

For issues or questions:
1. Check the [Ollama documentation](https://github.com/ollama/ollama)
2. Review error messages in the API responses
3. Check server logs for detailed error information
