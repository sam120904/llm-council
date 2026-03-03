"""OpenRouter API client for making LLM requests."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL

# Retry settings for free-tier rate limits
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
# Delay between launching each parallel request to avoid bursting rate limits
STAGGER_DELAY_SECONDS = 1.5


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API with retry on rate-limit (429).

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
                )

                # Retry on rate-limit
                if response.status_code == 429:
                    backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(f"[WARN] Model {model} rate-limited (429). Retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})...")
                    await asyncio.sleep(backoff)
                    continue

                if response.status_code != 200:
                    print(f"[ERROR] Model {model} returned HTTP {response.status_code}")
                    print(f"[ERROR] Response body: {response.text}")
                    return None

                data = response.json()

                if 'error' in data:
                    error_info = data['error']
                    # Some 429s come back as HTTP 200 with an error object
                    error_code = error_info.get('code') if isinstance(error_info, dict) else None
                    if error_code == 429:
                        backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
                        print(f"[WARN] Model {model} rate-limited (error body). Retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})...")
                        await asyncio.sleep(backoff)
                        continue
                    print(f"[ERROR] Model {model} API error: {error_info}")
                    return None

                if not data.get('choices'):
                    print(f"[ERROR] Model {model} returned no choices: {data}")
                    return None

                message = data['choices'][0]['message']

                return {
                    'content': message.get('content'),
                    'reasoning_details': message.get('reasoning_details')
                }

        except httpx.TimeoutException:
            print(f"[ERROR] Model {model} timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"[ERROR] Model {model} exception: {type(e).__name__}: {e}")
            return None

    # Exhausted all retries
    print(f"[ERROR] Model {model} failed after {MAX_RETRIES} retries (rate-limited)")
    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models with staggered launches to avoid rate-limit bursts.

    Each request is launched STAGGER_DELAY_SECONDS apart, then all run concurrently.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    # Launch tasks with a small stagger between each
    tasks = []
    for i, model in enumerate(models):
        if i > 0:
            await asyncio.sleep(STAGGER_DELAY_SECONDS)
        tasks.append(asyncio.create_task(query_model(model, messages)))

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
