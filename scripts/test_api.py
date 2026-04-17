#!/usr/bin/env python3
"""
Test script for MindScope API endpoints.
Run this after starting the backend services:
  docker-compose up -d
  cd apps/api && uvicorn mindscope_api.main:app --reload
  python ../../scripts/test_api.py
"""

import asyncio
import httpx
import json
import time


API_URL = "http://localhost:8000"


async def test_health():
    """Test health check endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print("✓ Health check passed")


async def test_root():
    """Test root endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print("✓ Root endpoint passed")


async def test_demo_job():
    """Test demo job submission and status tracking."""
    async with httpx.AsyncClient() as client:
        # Submit demo job
        response = await client.post(f"{API_URL}/jobs/demo", json={})
        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"✓ Demo job submitted: {job_id}")

        # Check status (immediately may be queued or processing)
        status_response = await client.get(f"{API_URL}/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["job_id"] == job_id
        print(f"  Job status: {status['status']}")


async def test_custom_job():
    """Test custom job submission."""
    async with httpx.AsyncClient() as client:
        payload = {
            "video_path": "/tmp/test_video.mp4",
            "audio_path": "/tmp/test_audio.wav",
            "text": "Test narration",
        }

        response = await client.post(f"{API_URL}/jobs", json=payload)
        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["job_id"]
        assert job_data["status"] == "queued"
        print(f"✓ Custom job submitted: {job_id}")


async def main():
    """Run all tests."""
    print("Testing MindScope API...\n")

    try:
        await test_health()
        await test_root()
        await test_demo_job()
        await test_custom_job()

        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
