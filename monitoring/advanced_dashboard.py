#!/usr/bin/env python3
"""
Advanced ML API Monitoring Dashboard with Alerting
Provides real-time monitoring with SLA tracking and alerts.
"""

import requests
import time
import re
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class Alert:
    """Represents a monitoring alert."""

    level: str  # INFO, WARNING, CRITICAL
    message: str
    timestamp: datetime
    metric: str
    value: float
    threshold: float


class SLATracker:
    """Tracks SLA metrics and generates alerts."""

    def __init__(self):
        self.thresholds = {
            "latency_p95_ms": 1000,  # 95th percentile latency
            "error_rate_percent": 5,  # Error rate percentage
            "availability_percent": 99.9,  # Availability percentage
            "prediction_latency_ms": 500,  # Prediction endpoint latency
        }

        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        self.metrics_history = defaultdict(
            lambda: deque(maxlen=60)
        )  # 10 minutes of history

    def check_alerts(self, metrics: Dict) -> List[Alert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []

        # Check prediction endpoint latency
        for endpoint, data in metrics.get("latency_data", {}).items():
            if "POST /predict" in endpoint and "avg_latency_ms" in data:
                latency = data["avg_latency_ms"]
                if latency > self.thresholds["prediction_latency_ms"]:
                    alert = Alert(
                        level="WARNING" if latency < 1000 else "CRITICAL",
                        message=f"High prediction latency: {latency:.1f}ms",
                        timestamp=datetime.now(),
                        metric="prediction_latency",
                        value=latency,
                        threshold=self.thresholds["prediction_latency_ms"],
                    )
                    new_alerts.append(alert)

        # Check error rates
        total_requests = sum(metrics.get("request_counts", {}).values())
        error_requests = sum(
            count
            for endpoint, count in metrics.get("request_counts", {}).items()
            if "4xx" in endpoint or "5xx" in endpoint
        )

        if total_requests > 0:
            error_rate = (error_requests / total_requests) * 100
            if error_rate > self.thresholds["error_rate_percent"]:
                alert = Alert(
                    level="CRITICAL",
                    message=f"High error rate: {error_rate:.1f}%",
                    timestamp=datetime.now(),
                    metric="error_rate",
                    value=error_rate,
                    threshold=self.thresholds["error_rate_percent"],
                )
                new_alerts.append(alert)

        # Store alerts
        for alert in new_alerts:
            self.alerts.append(alert)

        return new_alerts


class AdvancedMetricsCollector:
    """Advanced metrics collector with SLA tracking."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.metrics_url = f"{api_url}/metrics"
        self.sla_tracker = SLATracker()
        self.start_time = datetime.now()

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
            "uptime": str(datetime.now() - self.start_time).split(".")[0],
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

    def calculate_sla_metrics(self, metrics: Dict) -> Dict:
        """Calculate SLA metrics."""
        sla_metrics = {}

        # Calculate total requests and errors
        total_requests = sum(metrics.get("request_counts", {}).values())
        error_requests = sum(
            count
            for endpoint, count in metrics.get("request_counts", {}).items()
            if "4xx" in endpoint or "5xx" in endpoint
        )

        # Error rate
        sla_metrics["error_rate_percent"] = (
            (error_requests / total_requests * 100) if total_requests > 0 else 0
        )

        # Availability (simplified - based on successful requests)
        success_requests = total_requests - error_requests
        sla_metrics["availability_percent"] = (
            (success_requests / total_requests * 100) if total_requests > 0 else 100
        )

        # Average latency across all endpoints
        latencies = [
            data.get("avg_latency_ms", 0)
            for data in metrics.get("latency_data", {}).values()
        ]
        sla_metrics["avg_latency_ms"] = (
            sum(latencies) / len(latencies) if latencies else 0
        )

        # Prediction endpoint specific metrics
        for endpoint, data in metrics.get("latency_data", {}).items():
            if "POST /predict" in endpoint:
                sla_metrics["prediction_latency_ms"] = data.get("avg_latency_ms", 0)
                sla_metrics["prediction_requests"] = data.get("count", 0)

        return sla_metrics

    def display_dashboard(self, metrics: Dict, sla_metrics: Dict, alerts: List[Alert]):
        """Display comprehensive dashboard."""
        print("\n" + "=" * 100)
        print(f"🚀 ML API MONITORING DASHBOARD - {metrics['timestamp']}")
        print(f"⏰ Uptime: {metrics['uptime']}")
        print("=" * 100)

        # SLA Status
        print("\n📊 SLA STATUS:")
        print("-" * 50)

        # Availability
        availability = sla_metrics.get("availability_percent", 100)
        avail_status = (
            "🟢" if availability >= 99.9 else "🟡" if availability >= 99 else "🔴"
        )
        print(
            f"  Availability:        {avail_status} {availability:>6.2f}% (SLA: 99.9%)"
        )

        # Error Rate
        error_rate = sla_metrics.get("error_rate_percent", 0)
        error_status = "🟢" if error_rate <= 1 else "🟡" if error_rate <= 5 else "🔴"
        print(f"  Error Rate:          {error_status} {error_rate:>6.2f}% (SLA: <5%)")

        # Average Latency
        avg_latency = sla_metrics.get("avg_latency_ms", 0)
        latency_status = (
            "🟢" if avg_latency <= 200 else "🟡" if avg_latency <= 500 else "🔴"
        )
        print(f"  Avg Latency:         {latency_status} {avg_latency:>6.1f}ms")

        # Prediction Latency
        pred_latency = sla_metrics.get("prediction_latency_ms", 0)
        pred_status = (
            "🟢" if pred_latency <= 500 else "🟡" if pred_latency <= 1000 else "🔴"
        )
        print(
            f"  Prediction Latency:  {pred_status} "
            f"{pred_latency:>6.1f}ms (SLA: <500ms)"
        )

        # Request Counts
        print("\n🔢 REQUEST METRICS:")
        print("-" * 50)
        for endpoint, count in metrics.get("request_counts", {}).items():
            status_icon = (
                "🟢" if "2xx" in endpoint else "🟡" if "4xx" in endpoint else "🔴"
            )
            print(f"  {endpoint:<45} {status_icon} {int(count):>5}")

        # Latency Breakdown
        print("\n⏱️  LATENCY BREAKDOWN:")
        print("-" * 50)
        for endpoint, data in metrics.get("latency_data", {}).items():
            if "avg_latency_ms" in data:
                latency = data["avg_latency_ms"]
                status = "🟢" if latency < 100 else "🟡" if latency < 500 else "🔴"
                requests = int(data.get("count", 0))
                print(f"  {endpoint:<35} {status} {latency:>6.1f}ms ({requests} reqs)")

        # Active Alerts
        if alerts:
            print(f"\n🚨 ACTIVE ALERTS ({len(alerts)}):")
            print("-" * 50)
            for alert in alerts[-5:]:  # Show last 5 alerts
                icon = (
                    "🔴"
                    if alert.level == "CRITICAL"
                    else "🟡" if alert.level == "WARNING" else "🔵"
                )
                time_str = alert.timestamp.strftime("%H:%M:%S")
                print(f"  {icon} [{time_str}] {alert.level}: {alert.message}")

        # System Info
        print("\n🐍 SYSTEM INFO:")
        print("-" * 50)
        for key, value in metrics.get("system_info", {}).items():
            print(f"  {key:<20} {value}")

        print("\n" + "=" * 100)

    def monitor_with_sla(self, interval: int = 10):
        """Monitor with SLA tracking and alerting."""
        print("🚀 Starting Advanced ML API Monitoring...")
        print(f"📡 Fetching metrics every {interval} seconds")
        print("🔔 SLA monitoring and alerting enabled")
        print("🛑 Press Ctrl+C to stop")

        try:
            while True:
                metrics_text = self.fetch_metrics()
                if metrics_text:
                    metrics = self.parse_metrics(metrics_text)
                    sla_metrics = self.calculate_sla_metrics(metrics)
                    alerts = self.sla_tracker.check_alerts(metrics)

                    self.display_dashboard(
                        metrics, sla_metrics, list(self.sla_tracker.alerts)
                    )

                    # Print new alerts
                    for alert in alerts:
                        icon = "🔴" if alert.level == "CRITICAL" else "🟡"
                        print(f"\n{icon} NEW ALERT: {alert.message}")
                else:
                    print(f"❌ Failed to fetch metrics at {datetime.now()}")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

            # Summary report
            print("\n📋 MONITORING SUMMARY:")
            print("-" * 50)
            print(f"Total monitoring time: {datetime.now() - self.start_time}")
            print(f"Total alerts generated: {len(self.sla_tracker.alerts)}")

            if self.sla_tracker.alerts:
                critical_alerts = sum(
                    1 for alert in self.sla_tracker.alerts if alert.level == "CRITICAL"
                )
                warning_alerts = sum(
                    1 for alert in self.sla_tracker.alerts if alert.level == "WARNING"
                )
                print(f"Critical alerts: {critical_alerts}")
                print(f"Warning alerts: {warning_alerts}")


def main():
    """Main function to run the advanced monitoring dashboard."""
    collector = AdvancedMetricsCollector()

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

    # Start monitoring
    collector.monitor_with_sla()


if __name__ == "__main__":
    main()
