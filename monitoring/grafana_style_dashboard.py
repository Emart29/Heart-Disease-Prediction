#!/usr/bin/env python3
"""
Grafana-style Dashboard for ML API Monitoring
Creates visual charts and graphs using text-based plotting.
"""

import requests
import time
import re
from datetime import datetime
from typing import Dict, List
from collections import defaultdict, deque
import statistics


class TextChart:
    """Creates text-based charts similar to Grafana."""

    @staticmethod
    def create_line_chart(
        data: List[float], title: str, width: int = 60, height: int = 10
    ) -> str:
        """Create a text-based line chart."""
        if not data or len(data) < 2:
            return f"{title}\n" + "No data available"

        # Normalize data to chart height
        min_val = min(data)
        max_val = max(data)

        if max_val == min_val:
            normalized = [height // 2] * len(data)
        else:
            normalized = [
                int((val - min_val) / (max_val - min_val) * (height - 1))
                for val in data
            ]

        # Create chart
        chart_lines = []

        # Title and scale
        chart_lines.append(f"📊 {title}")
        chart_lines.append(
            f"Max: {max_val:.2f} {'─' * (width - 20)} Min: {min_val:.2f}"
        )

        # Chart body
        for y in range(height - 1, -1, -1):
            line = ""
            for x in range(len(normalized)):
                if x < len(data):
                    if normalized[x] == y:
                        line += "●"
                    elif normalized[x] > y:
                        line += "│"
                    else:
                        line += " "
                else:
                    line += " "

            # Add y-axis labels
            if y == height - 1:
                line += f" │ {max_val:.1f}"
            elif y == 0:
                line += f" │ {min_val:.1f}"
            elif y == height // 2:
                line += f" │ {(max_val + min_val) / 2:.1f}"
            else:
                line += " │"

            chart_lines.append(line)

        # X-axis
        chart_lines.append("─" * len(normalized) + "─┘")

        return "\n".join(chart_lines)

    @staticmethod
    def create_bar_chart(data: Dict[str, float], title: str, width: int = 50) -> str:
        """Create a text-based bar chart."""
        if not data:
            return f"{title}\nNo data available"

        chart_lines = [f"📊 {title}"]

        max_val = max(data.values()) if data.values() else 1
        max_label_len = max(len(label) for label in data.keys()) if data else 0

        for label, value in data.items():
            bar_length = int((value / max_val) * width) if max_val > 0 else 0
            bar = "█" * bar_length
            padding = " " * (max_label_len - len(label))
            chart_lines.append(f"{label}{padding} │{bar} {value:.2f}")

        return "\n".join(chart_lines)

    @staticmethod
    def create_gauge(
        value: float, max_value: float, title: str, thresholds: Dict[str, float] = None
    ) -> str:
        """Create a text-based gauge."""
        if thresholds is None:
            thresholds = {"good": 0.7, "warning": 0.9}

        percentage = (value / max_value) * 100 if max_value > 0 else 0

        # Determine color/status
        if percentage <= thresholds["good"] * 100:
            status = "🟢 GOOD"
        elif percentage <= thresholds["warning"] * 100:
            status = "🟡 WARNING"
        else:
            status = "🔴 CRITICAL"

        # Create gauge visualization
        gauge_width = 30
        filled = int((percentage / 100) * gauge_width)
        empty = gauge_width - filled

        gauge_bar = "█" * filled + "░" * empty

        return (
            f"🎯 {title}\n[{gauge_bar}] {percentage:.1f}% {status}\n"
            f"{value:.2f} / {max_value:.2f}"
        )


class GrafanaStyleDashboard:
    """Creates a comprehensive dashboard similar to Grafana."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.metrics_url = f"{api_url}/metrics"
        self.metrics_history = defaultdict(
            lambda: deque(maxlen=60)
        )  # 10 minutes of history
        self.start_time = datetime.now()

    def fetch_and_parse_metrics(self) -> Dict:
        """Fetch and parse all metrics."""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            return self.parse_comprehensive_metrics(response.text)
        except requests.RequestException as e:
            print(f"Error fetching metrics: {e}")
            return {}

    def parse_comprehensive_metrics(self, metrics_text: str) -> Dict:
        """Parse all available metrics."""
        metrics = {
            "timestamp": datetime.now(),
            "request_counts": {},
            "latency_data": {},
            "response_sizes": {},
            "business_metrics": {},
            "system_metrics": {},
        }

        lines = metrics_text.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # HTTP Request metrics
            if "http_requests_total{" in line:
                match = re.search(
                    r'handler="([^"]+)".*method="([^"]+)".*'
                    r'status="([^"]+)".*} (\d+\.?\d*)',
                    line,
                )
                if match:
                    handler, method, status, count = match.groups()
                    key = f"{method} {handler}"
                    if key not in metrics["request_counts"]:
                        metrics["request_counts"][key] = {
                            "total": 0,
                            "2xx": 0,
                            "4xx": 0,
                            "5xx": 0,
                        }

                    metrics["request_counts"][key]["total"] += float(count)
                    if "2xx" in status:
                        metrics["request_counts"][key]["2xx"] += float(count)
                    elif "4xx" in status:
                        metrics["request_counts"][key]["4xx"] += float(count)
                    elif "5xx" in status:
                        metrics["request_counts"][key]["5xx"] += float(count)

            # Latency metrics
            elif "http_request_duration_seconds_sum{" in line:
                match = re.search(
                    r'handler="([^"]+)".*method="([^"]+)".*} (\d+\.?\d*)', line
                )
                if match:
                    handler, method, total_time = match.groups()
                    key = f"{method} {handler}"
                    if key not in metrics["latency_data"]:
                        metrics["latency_data"][key] = {}
                    metrics["latency_data"][key]["total_time"] = float(total_time)

            elif "http_request_duration_seconds_count{" in line:
                match = re.search(
                    r'handler="([^"]+)".*method="([^"]+)".*} (\d+\.?\d*)', line
                )
                if match:
                    handler, method, count = match.groups()
                    key = f"{method} {handler}"
                    if key not in metrics["latency_data"]:
                        metrics["latency_data"][key] = {}
                    metrics["latency_data"][key]["count"] = float(count)

            # Business metrics
            elif "ml_predictions_total{" in line:
                if 'prediction_result="positive"' in line:
                    match = re.search(r"} (\d+\.?\d*)", line)
                    if match:
                        metrics["business_metrics"]["positive_predictions"] = float(
                            match.group(1)
                        )
                elif 'prediction_result="negative"' in line:
                    match = re.search(r"} (\d+\.?\d*)", line)
                    if match:
                        metrics["business_metrics"]["negative_predictions"] = float(
                            match.group(1)
                        )

            elif "ml_model_accuracy" in line and not line.startswith("#"):
                match = re.search(r"} (\d+\.?\d*)", line)
                if match:
                    metrics["business_metrics"]["model_accuracy"] = float(
                        match.group(1)
                    )

            elif "ml_prediction_confidence_sum" in line:
                match = re.search(r"} (\d+\.?\d*)", line)
                if match:
                    metrics["business_metrics"]["confidence_sum"] = float(
                        match.group(1)
                    )

            elif "ml_prediction_confidence_count" in line:
                match = re.search(r"} (\d+\.?\d*)", line)
                if match:
                    count = float(match.group(1))
                    if count > 0 and "confidence_sum" in metrics["business_metrics"]:
                        metrics["business_metrics"]["avg_confidence"] = (
                            metrics["business_metrics"]["confidence_sum"] / count
                        )

            # System metrics
            elif "python_info{" in line:
                match = re.search(r'version="([^"]+)"', line)
                if match:
                    metrics["system_metrics"]["python_version"] = match.group(1)

        # Calculate derived metrics
        for endpoint, data in metrics["latency_data"].items():
            if "total_time" in data and "count" in data and data["count"] > 0:
                data["avg_latency_ms"] = (data["total_time"] / data["count"]) * 1000

        return metrics

    def update_history(self, metrics: Dict):
        """Update metrics history for trending."""
        # Store key metrics for trending
        total_requests = sum(
            data["total"] for data in metrics.get("request_counts", {}).values()
        )
        total_errors = sum(
            data.get("4xx", 0) + data.get("5xx", 0)
            for data in metrics.get("request_counts", {}).values()
        )

        self.metrics_history["total_requests"].append(total_requests)
        self.metrics_history["total_errors"].append(total_errors)
        self.metrics_history["error_rate"].append(
            (total_errors / total_requests * 100) if total_requests > 0 else 0
        )

        # Latency trending
        latencies = [
            data.get("avg_latency_ms", 0)
            for data in metrics.get("latency_data", {}).values()
        ]
        avg_latency = statistics.mean(latencies) if latencies else 0
        self.metrics_history["avg_latency"].append(avg_latency)

        # Business metrics trending
        model_accuracy = metrics.get("business_metrics", {}).get("model_accuracy", 0)
        avg_confidence = metrics.get("business_metrics", {}).get("avg_confidence", 0)

        self.metrics_history["model_accuracy"].append(model_accuracy)
        self.metrics_history["avg_confidence"].append(avg_confidence)

    def create_dashboard(self, metrics: Dict) -> str:
        """Create a comprehensive Grafana-style dashboard."""
        dashboard_lines = []

        # Header
        dashboard_lines.append("=" * 120)
        dashboard_lines.append("🎛️  GRAFANA-STYLE ML API DASHBOARD")
        dashboard_lines.append(
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"⏱️  Uptime: {str(datetime.now() - self.start_time).split('.')[0]}"
        )
        dashboard_lines.append("=" * 120)

        # Row 1: Key Performance Indicators
        dashboard_lines.append("\n📊 KEY PERFORMANCE INDICATORS")
        dashboard_lines.append("-" * 50)

        # Calculate KPIs
        total_requests = sum(
            data["total"] for data in metrics.get("request_counts", {}).values()
        )
        total_errors = sum(
            data.get("4xx", 0) + data.get("5xx", 0)
            for data in metrics.get("request_counts", {}).values()
        )
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        latencies = [
            data.get("avg_latency_ms", 0)
            for data in metrics.get("latency_data", {}).values()
        ]
        avg_latency = statistics.mean(latencies) if latencies else 0

        # KPI Gauges
        chart = TextChart()

        # Error Rate Gauge
        error_gauge = chart.create_gauge(
            error_rate, 10, "Error Rate (%)", {"good": 0.1, "warning": 0.5}
        )

        # Latency Gauge
        latency_gauge = chart.create_gauge(
            avg_latency, 1000, "Avg Latency (ms)", {"good": 0.2, "warning": 0.5}
        )

        # Model Accuracy Gauge
        model_accuracy = (
            metrics.get("business_metrics", {}).get("model_accuracy", 0) * 100
        )
        accuracy_gauge = chart.create_gauge(
            model_accuracy, 100, "Model Accuracy (%)", {"good": 0.8, "warning": 0.9}
        )

        # Display gauges side by side (simplified)
        dashboard_lines.append(error_gauge)
        dashboard_lines.append("")
        dashboard_lines.append(latency_gauge)
        dashboard_lines.append("")
        dashboard_lines.append(accuracy_gauge)

        # Row 2: Request Rate Trending
        dashboard_lines.append("\n\n📈 REQUEST RATE TRENDING")
        dashboard_lines.append("-" * 50)

        if len(self.metrics_history["total_requests"]) > 1:
            request_chart = chart.create_line_chart(
                list(self.metrics_history["total_requests"]), "Total Requests Over Time"
            )
            dashboard_lines.append(request_chart)
        else:
            dashboard_lines.append("Collecting data... (need at least 2 data points)")

        # Row 3: Error Rate Trending
        dashboard_lines.append("\n\n🚨 ERROR RATE TRENDING")
        dashboard_lines.append("-" * 50)

        if len(self.metrics_history["error_rate"]) > 1:
            error_chart = chart.create_line_chart(
                list(self.metrics_history["error_rate"]), "Error Rate (%) Over Time"
            )
            dashboard_lines.append(error_chart)
        else:
            dashboard_lines.append("Collecting data... (need at least 2 data points)")

        # Row 4: Latency Distribution
        dashboard_lines.append("\n\n⏱️  LATENCY DISTRIBUTION")
        dashboard_lines.append("-" * 50)

        latency_by_endpoint = {}
        for endpoint, data in metrics.get("latency_data", {}).items():
            if "avg_latency_ms" in data:
                latency_by_endpoint[
                    endpoint.replace("GET ", "").replace("POST ", "")
                ] = data["avg_latency_ms"]

        if latency_by_endpoint:
            latency_bar_chart = chart.create_bar_chart(
                latency_by_endpoint, "Average Latency by Endpoint (ms)"
            )
            dashboard_lines.append(latency_bar_chart)
        else:
            dashboard_lines.append("No latency data available")

        # Row 5: Business Metrics
        dashboard_lines.append("\n\n🏥 BUSINESS METRICS")
        dashboard_lines.append("-" * 50)

        business_data = {}
        pos_preds = metrics.get("business_metrics", {}).get("positive_predictions", 0)
        neg_preds = metrics.get("business_metrics", {}).get("negative_predictions", 0)

        if pos_preds > 0 or neg_preds > 0:
            business_data["Positive"] = pos_preds
            business_data["Negative"] = neg_preds

            business_chart = chart.create_bar_chart(
                business_data, "Prediction Distribution"
            )
            dashboard_lines.append(business_chart)
        else:
            dashboard_lines.append("No prediction data available")

        # Row 6: System Status
        dashboard_lines.append("\n\n🖥️  SYSTEM STATUS")
        dashboard_lines.append("-" * 50)

        # System info table
        system_info = [
            f"Python Version: "
            f"{metrics.get('system_metrics', {}).get('python_version', 'Unknown')}",
            f"Total Requests: {int(total_requests):,}",
            f"Total Errors: {int(total_errors):,}",
            f"Success Rate: {100 - error_rate:.2f}%",
            f"Avg Response Time: {avg_latency:.2f}ms",
        ]

        for info in system_info:
            dashboard_lines.append(f"  {info}")

        # Footer
        dashboard_lines.append("\n" + "=" * 120)
        dashboard_lines.append(
            "🔄 Dashboard updates every 15 seconds | 🛑 Press Ctrl+C to stop"
        )
        dashboard_lines.append("=" * 120)

        return "\n".join(dashboard_lines)

    def run_dashboard(self, refresh_interval: int = 15):
        """Run the live dashboard."""
        print("🎛️  Starting Grafana-Style Dashboard...")
        print(f"🔄 Refreshing every {refresh_interval} seconds")
        print("🛑 Press Ctrl+C to stop")

        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")

                # Fetch and display metrics
                metrics = self.fetch_and_parse_metrics()
                if metrics:
                    self.update_history(metrics)
                    dashboard = self.create_dashboard(metrics)
                    print(dashboard)
                else:
                    print("❌ Failed to fetch metrics")

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")


def main():
    """Main function."""
    dashboard = GrafanaStyleDashboard()

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

    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
