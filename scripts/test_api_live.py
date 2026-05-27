"""
Live end-to-end test — requires the API server to be running.

Start the server first:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Then run:
    python scripts/test_api_live.py
"""

import asyncio
import time

import httpx

BASE_URL = "http://localhost:8000"


async def run_live_test():
    async with httpx.AsyncClient(timeout=300) as client:

        # Health check first
        r = await client.get(f"{BASE_URL}/health")
        assert r.status_code == 200, f"Health check failed: {r.text}"
        print(f"✅ Health check passed: {r.json()}")

        # Start research
        payload = {
            "query": "How does in-context learning work in large language models?",
            "max_sections": 2,
            "follow_guidelines": False,
        }
        r = await client.post(f"{BASE_URL}/research", json=payload)
        assert r.status_code == 202, f"Task creation failed: {r.text}"

        task_id = r.json()["task_id"]
        print(f"\n🔍 Task created: {task_id}")
        print(f"   Polling {BASE_URL}/research/{task_id}...")

        # Poll until complete
        start = time.time()
        last_status = None

        while True:
            r = await client.get(f"{BASE_URL}/research/{task_id}")
            data = r.json()
            status = data["status"]

            if status != last_status:
                print(f"   Status: {status} ({time.time()-start:.0f}s elapsed)")
                last_status = status

            if status == "complete":
                print(f"\n✅ COMPLETE in {data['elapsed_seconds']:.1f}s")
                print(f"   Sources: {len(data['sources'])}")
                print(f"   Sections: {data['sections']}")
                print(f"   Report preview:")
                print("   " + data["report"][:300].replace("\n", "\n   "))
                break

            if status == "failed":
                print(f"\n❌ FAILED: {data['error']}")
                break

            if time.time() - start > 300:
                print("\n⏰ Timeout after 5 minutes")
                break

            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(run_live_test())
