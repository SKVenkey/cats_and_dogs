"""
Smoke test for the deployed inference service.
Calls /health and /predict endpoints. Exits with code 1 on failure.

Usage:
    python scripts/smoke_test.py --host http://localhost:8000
"""

import argparse
import sys
import os
import io
import json
import time

import requests
from PIL import Image
import numpy as np


def wait_for_service(host: str, max_wait: int = 60):
    """Poll the health endpoint until the service is ready."""
    url = f"{host}/health"
    for i in range(max_wait):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"Service healthy after {i+1}s")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print(f"Service not ready after {max_wait}s")
    return False


def create_dummy_image() -> bytes:
    """Create a synthetic RGB image for testing."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health(host: str):
    print(f"\n[1/2] Testing GET {host}/health ...")
    r = requests.get(f"{host}/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    body = r.json()
    assert body.get("status") == "ok", f"Unexpected health response: {body}"
    print(f"    ✓ Health OK: {body}")


def test_predict(host: str):
    print(f"\n[2/2] Testing POST {host}/predict ...")
    img_bytes = create_dummy_image()
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    r = requests.post(f"{host}/predict", files=files, timeout=30)
    assert r.status_code == 200, f"Predict failed: {r.status_code} {r.text}"
    body = r.json()
    assert "label" in body, f"Missing 'label' in response: {body}"
    assert "probabilities" in body, f"Missing 'probabilities' in response: {body}"
    assert "confidence" in body, f"Missing 'confidence' in response: {body}"
    print(f"    ✓ Prediction OK: label={body['label']}, confidence={body['confidence']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--wait", type=int, default=60, help="Max seconds to wait for service")
    args = parser.parse_args()

    if not wait_for_service(args.host, args.wait):
        sys.exit(1)

    try:
        test_health(args.host)
        test_predict(args.host)
        print("\n✅ All smoke tests passed!")
    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
