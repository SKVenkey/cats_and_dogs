"""
M5: Post-deployment model performance monitoring.
Simulates a batch of real requests, logs results, and computes drift metrics.

Usage:
    python scripts/monitor.py --host http://localhost:8000 --n_samples 50
"""

import argparse
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def generate_simulated_sample(label: str) -> bytes:
    """Create a synthetic image that loosely resembles the given class."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_monitoring(host: str, n_samples: int, log_file: str):
    results = []
    labels = ["Cat", "Dog"]
    correct = 0

    print(f"Running {n_samples} simulated inference calls against {host}...")

    for i in range(n_samples):
        true_label = random.choice(labels)
        img_bytes = generate_simulated_sample(true_label)

        start = time.time()
        try:
            r = requests.post(
                f"{host}/predict",
                files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
                timeout=30,
            )
            latency = time.time() - start
            body = r.json()
            pred_label = body.get("label", "unknown")
            confidence = body.get("confidence", 0.0)
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1

            record = {
                "sample_id": i,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "latency_s": round(latency, 4),
                "correct": is_correct,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            record = {"sample_id": i, "error": str(e)}

        results.append(record)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    # ── Summary ───────────────────────────────────────────────────────────────
    accuracy = correct / n_samples
    latencies = [r["latency_s"] for r in results if "latency_s" in r]
    avg_latency = np.mean(latencies) if latencies else 0.0
    p95_latency = np.percentile(latencies, 95) if latencies else 0.0

    summary = {
        "total_samples": n_samples,
        "accuracy": round(accuracy, 4),
        "avg_latency_s": round(avg_latency, 4),
        "p95_latency_s": round(p95_latency, 4),
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"\n{'─'*50}")
    print(f"  Accuracy      : {accuracy:.2%}")
    print(f"  Avg Latency   : {avg_latency*1000:.1f} ms")
    print(f"  P95 Latency   : {p95_latency*1000:.1f} ms")
    print(f"{'─'*50}")

    # Save log
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nMonitoring log saved to {log_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--log_file", default="artifacts/monitoring_log.json")
    args = parser.parse_args()
    run_monitoring(args.host, args.n_samples, args.log_file)


if __name__ == "__main__":
    main()
