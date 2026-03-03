"""Quick diagnostic: test each free model individually."""
import asyncio
import httpx
from backend.config import OPENROUTER_API_KEY, OPENROUTER_API_URL, COUNCIL_MODELS, CHAIRMAN_MODEL

MODELS_TO_TEST = COUNCIL_MODELS + [CHAIRMAN_MODEL]

async def test_model(model: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            print(f"\n{'='*60}")
            print(f"Model: {model}")
            print(f"HTTP Status: {resp.status_code}")
            data = resp.json()
            if "error" in data:
                print(f"ERROR: {data['error']}")
            elif data.get("choices"):
                content = data["choices"][0]["message"].get("content", "")
                print(f"SUCCESS: {content[:100]}")
            else:
                print(f"UNEXPECTED: {data}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"EXCEPTION: {type(e).__name__}: {e}")

async def main():
    print("Testing models one at a time with 3s gaps...\n")
    for model in MODELS_TO_TEST:
        await test_model(model)
        await asyncio.sleep(3)

asyncio.run(main())
