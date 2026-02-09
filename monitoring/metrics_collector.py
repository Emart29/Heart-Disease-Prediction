#!/usr/bin/env python3
"""
Simple metrics collector and dashboard for Heart Disease Prediction API.
This script collects metrics from the /metrics endpoint and displays them.
"""

import requests
import time
import re
from datetime import datetime
from typing import Dict


class MetricsCollector:
    """Collects and parses Prometheus metrics from FastAPI."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.metrics_url = f"{api_url}/metrics"

    def fetch_metrics(self) -> str:
        """Fetch raw metrics from the API."""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching metrics: {e}")
            return ""

    def parse_metrics(self, metrics_text: str) -> Dict:
        """Parse Prometheus metrics into a structured format."""
        parsed = {
            "timestamp": datetime.now().isoformat(),
            "request_counts": {},
            "latency_data": {},
            "response_sizes": {},
            "system_info": {},
        }

        lines = metrics_text.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Parse request counts
            if "http_requests_total{" in line:
                match = re.search(
                    r'handler="([^"]+)".*method="([^"]+)".*'
                    r'status="([^"]+)".*} (\d+\.?\d*)',
                    line,
                )
                if match:
                    handler, method, status, count = match.groups()
                    key = f"{method} {handler} ({status})"
                    parsed["request_counts"][key] = float(count)

            # Parse latency data
            elif "http_request_duration_seconds_sum{" in line:
                match = re.search(
                    r'handler="([^"]+)".*method="([^"]+)".*} (\d+\.?\d*)', line
                )
                if match:
                    handler, method, total_time = match.groups()
                    key = f"{method} {handler}"
                    if key not in parsed["latency_data"]:
                        parsed["latency_data"][key] = {}
                    parsed["latency_data"][key]["total_time"] = float(total_time)

            elif "http_request_duration_seconds_count{" in line:
                match = re.search(
                    r'handler="([^"]+)".*method="([^"]+)".*} (\d+\.?\d*)', line
                )
                if match:
                    handler, method, count = match.groups()
                    key = f"{method} {handler}"
                    if key not in parsed["latency_data"]:
                        parsed["latency_data"][key] = {}
                    parsed["latency_data"][key]["count"] = float(count)

            # Parse response sizes
            elif "http_response_size_bytes_sum{" in line:
                match = re.search(r'handler="([^"]+)".*} (\d+\.?\d*)', line)
                if match:
                    handler, size = match.groups()
                    parsed["response_sizes"][handler] = float(size)

            # Parse Python info
            elif "python_info{" in line:
                match = re.search(r'version="([^"]+)".*} (\d+\.?\d*)', line)
                if match:
                    version, _ = match.groups()
                    parsed["system_info"]["python_version"] = version

        # Calculate average latencies
        for endpoint, data in parsed["latency_data"].items():
            if "total_time" in data and "count" in data and data["count"] > 0:
                data["avg_latency_ms"] = (data["total_time"] / data["count"]) * 1000

        return parsed

    def display_metrics(self, metrics: Dict):
        """Display metrics in a readable format."""
        print("\n" + "=" * 80)
        print(f"📊 HEART DISEASE API METRICS - {metrics['timestamp']}")
        print("=" * 80)

        # Request Counts
        print("\n🔢 REQUEST COUNTS:")
        print("-" * 40)
        for endpoint, count in metrics["request_counts"].items():
            print(f"  {endpoint:<35} {int(count):>5}")

        # Latency Data
        print("\n⏱️  AVERAGE LATENCY:")
        print("-" * 40)
        for endpoint, data in metrics["latency_data"].items():
            if "avg_latency_ms" in data:
                latency = data["avg_latency_ms"]
                status = "🟢" if latency < 100 else "🟡" if latency < 500 else "🔴"
                print(f"  {endpoint:<35} {status} {latency:>6.1f}ms")

        # Response Sizes
        print("\n📦 RESPONSE SIZES:")
        print("-" * 40)
        for endpoint, size in metrics["response_sizes"].items():
            size_kb = size / 1024 if size > 1024 else size
            unit = "KB" if size > 1024 else "B"
            print(f"  {endpoint:<35} {size_kb:>6.1f}{unit}")

        # System Info
        if metrics["system_info"]:
            print("\n🐍 SYSTEM INFO:")
            print("-" * 40)
            for key, value in metrics["system_info"].items():
                print(f"  {key:<35} {value}")

        print("\n" + "=" * 80)

    def monitor_continuously(self, interval: int = 10):
        """Monitor metrics continuously."""
        print("🚀 Starting continuous monitoring...")
        print(f"📡 Fetching metrics every {interval} seconds")
        print("🛑 Press Ctrl+C to stop")

        try:
            while True:
                metrics_text = self.fetch_metrics()
                if metrics_text:
                    metrics = self.parse_metrics(metrics_text)
                    self.display_metrics(metrics)
                else:
                    print(f"❌ Failed to fetch metrics at {datetime.now()}")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")


def main():
    """Main function to run the metrics collector."""
    collector = MetricsCollector()

    print("🔍 Testing API connection...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running!")
        else:
            print(f"⚠️  API returned status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure the API is running on http://localhost:8000")
        return

    # Show current metrics once
    print("\n📊 Current Metrics Snapshot:")
    metrics_text = collector.fetch_metrics()
    if metrics_text:
        metrics = collector.parse_metrics(metrics_text)
        collector.display_metrics(metrics)

    # Ask if user wants continuous monitoring
    try:
        choice = input("\n🔄 Start continuous monitoring? (y/N): ").lower().strip()
        if choice in ["y", "yes"]:
            collector.monitor_continuously()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
