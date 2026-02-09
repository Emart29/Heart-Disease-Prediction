#!/usr/bin/env python3
"""
Business Metrics Dashboard for ML API
Tracks prediction accuracy, model performance, and business KPIs.
"""

import requests
import time
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import smtplib

try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False


@dataclass
class BusinessAlert:
    """Business-focused alert."""

    level: str
    message: str
    timestamp: datetime
    metric: str
    value: float
    threshold: float
    business_impact: str


class BusinessMetricsTracker:
    """Tracks business-specific ML metrics."""

    def __init__(self):
        self.thresholds = {
            "model_accuracy": 0.80,  # Model accuracy threshold
            "prediction_confidence_avg": 0.60,  # Average confidence threshold
            "high_risk_predictions_rate": 0.30,  # High risk prediction rate
            "low_confidence_rate": 0.20,  # Low confidence predictions rate
        }

        self.alerts = deque(maxlen=100)
        self.business_metrics_history = defaultdict(lambda: deque(maxlen=100))

    def check_business_alerts(self, metrics: Dict) -> List[BusinessAlert]:
        """Check business metrics and generate alerts."""
        new_alerts = []

        # Check model accuracy
        model_accuracy = metrics.get("business_metrics", {}).get("model_accuracy", 0)
        if model_accuracy > 0 and model_accuracy < self.thresholds["model_accuracy"]:
            alert = BusinessAlert(
                level="CRITICAL",
                message=f"Model accuracy below threshold: {model_accuracy:.3f}",
                timestamp=datetime.now(),
                metric="model_accuracy",
                value=model_accuracy,
                threshold=self.thresholds["model_accuracy"],
                business_impact=(
                    "Model may need retraining - affecting prediction quality"
                ),
            )
            new_alerts.append(alert)

        # Check prediction confidence
        avg_confidence = metrics.get("business_metrics", {}).get(
            "avg_prediction_confidence", 0
        )
        if (
            avg_confidence > 0
            and avg_confidence < self.thresholds["prediction_confidence_avg"]
        ):
            alert = BusinessAlert(
                level="WARNING",
                message=f"Low average prediction confidence: {avg_confidence:.3f}",
                timestamp=datetime.now(),
                metric="prediction_confidence",
                value=avg_confidence,
                threshold=self.thresholds["prediction_confidence_avg"],
                business_impact="Users may lose trust in predictions",
            )
            new_alerts.append(alert)

        # Check high-risk prediction rate
        high_risk_rate = metrics.get("business_metrics", {}).get("high_risk_rate", 0)
        if high_risk_rate > self.thresholds["high_risk_predictions_rate"]:
            alert = BusinessAlert(
                level="INFO",
                message=f"High rate of high-risk predictions: {high_risk_rate:.3f}",
                timestamp=datetime.now(),
                metric="high_risk_rate",
                value=high_risk_rate,
                threshold=self.thresholds["high_risk_predictions_rate"],
                business_impact="May indicate data drift or population health changes",
            )
            new_alerts.append(alert)

        for alert in new_alerts:
            self.alerts.append(alert)

        return new_alerts


class EmailAlerter:
    """Simple email alerting system."""

    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.enabled = False  # Set to True when configured

    def send_alert(
        self, alert: BusinessAlert, to_email: str, from_email: str, password: str
    ):
        """Send alert via email."""
        if not EMAIL_AVAILABLE:
            print(f"📧 EMAIL ALERT (email not available): {alert.message}")
            return

        if not self.enabled:
            print(f"📧 EMAIL ALERT (disabled): {alert.message}")
            return

        try:
            msg = MimeMultipart()
            msg["From"] = from_email
            msg["To"] = to_email
            msg["Subject"] = f"ML API Alert: {alert.level} - {alert.metric}"

            body = f"""
            Alert Level: {alert.level}
            Metric: {alert.metric}
            Current Value: {alert.value}
            Threshold: {alert.threshold}
            Business Impact: {alert.business_impact}
            Timestamp: {alert.timestamp}

            Message: {alert.message}
            """

            msg.attach(MimeText(body, "plain"))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(from_email, password)
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()

            print(f"📧 Email alert sent: {alert.message}")
        except Exception as e:
            print(f"❌ Failed to send email alert: {e}")


class SlackAlerter:
    """Simple Slack webhook alerting."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None

    def send_alert(self, alert: BusinessAlert):
        """Send alert to Slack."""
        if not self.enabled:
            print(f"💬 SLACK ALERT (disabled): {alert.message}")
            return

        try:
            color = (
                "#ff0000"
                if alert.level == "CRITICAL"
                else "#ffaa00" if alert.level == "WARNING" else "#0099ff"
            )

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"ML API Alert: {alert.level}",
                        "fields": [
                            {"title": "Metric", "value": alert.metric, "short": True},
                            {
                                "title": "Value",
                                "value": f"{alert.value:.3f}",
                                "short": True,
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.threshold:.3f}",
                                "short": True,
                            },
                            {
                                "title": "Business Impact",
                                "value": alert.business_impact,
                                "short": False,
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False,
                            },
                        ],
                        "timestamp": alert.timestamp.timestamp(),
                    }
                ]
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"💬 Slack alert sent: {alert.message}")
        except Exception as e:
            print(f"❌ Failed to send Slack alert: {e}")


class BusinessMetricsCollector:
    """Enhanced metrics collector with business focus."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.metrics_url = f"{api_url}/metrics"
        self.business_tracker = BusinessMetricsTracker()
        self.email_alerter = EmailAlerter()
        self.slack_alerter = SlackAlerter()  # Add webhook URL to enable
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

    def parse_business_metrics(self, metrics_text: str) -> Dict:
        """Parse business-specific metrics."""
        business_metrics = {}
        lines = metrics_text.split("\n")

        # Parse custom business metrics
        prediction_counts = {"positive": 0, "negative": 0}
        risk_level_counts = {"low": 0, "moderate": 0, "high": 0}

        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Parse prediction counts by result
            if "ml_predictions_total{" in line:
                if 'prediction_result="positive"' in line:
                    match = re.search(r"} (\d+\.?\d*)", line)
                    if match:
                        prediction_counts["positive"] += float(match.group(1))
                elif 'prediction_result="negative"' in line:
                    match = re.search(r"} (\d+\.?\d*)", line)
                    if match:
                        prediction_counts["negative"] += float(match.group(1))

                # Parse risk levels
                if 'risk_level="high"' in line:
                    match = re.search(r"} (\d+\.?\d*)", line)
                    if match:
                        risk_level_counts["high"] += float(match.group(1))
                elif 'risk_level="moderate"' in line:
                    match = re.search(r"} (\d+\.?\d*)", line)
                    if match:
                        risk_level_counts["moderate"] += float(match.group(1))
                elif 'risk_level="low"' in line:
                    match = re.search(r"} (\d+\.?\d*)", line)
                    if match:
                        risk_level_counts["low"] += float(match.group(1))

            # Parse model accuracy
            elif "ml_model_accuracy" in line and not line.startswith("#"):
                match = re.search(r"} (\d+\.?\d*)", line)
                if match:
                    business_metrics["model_accuracy"] = float(match.group(1))

            # Parse confidence histogram (simplified - get average from buckets)
            elif "ml_prediction_confidence_sum" in line:
                match = re.search(r"} (\d+\.?\d*)", line)
                if match:
                    confidence_sum = float(match.group(1))
            elif "ml_prediction_confidence_count" in line:
                match = re.search(r"} (\d+\.?\d*)", line)
                if match:
                    confidence_count = float(match.group(1))
                    if confidence_count > 0:
                        business_metrics["avg_prediction_confidence"] = (
                            confidence_sum / confidence_count
                        )

        # Calculate business KPIs
        total_predictions = sum(prediction_counts.values())
        total_risk_predictions = sum(risk_level_counts.values())

        if total_predictions > 0:
            business_metrics["positive_prediction_rate"] = (
                prediction_counts["positive"] / total_predictions
            )
            business_metrics["negative_prediction_rate"] = (
                prediction_counts["negative"] / total_predictions
            )

        if total_risk_predictions > 0:
            business_metrics["high_risk_rate"] = (
                risk_level_counts["high"] / total_risk_predictions
            )
            business_metrics["moderate_risk_rate"] = (
                risk_level_counts["moderate"] / total_risk_predictions
            )
            business_metrics["low_risk_rate"] = (
                risk_level_counts["low"] / total_risk_predictions
            )

        business_metrics["total_predictions"] = total_predictions
        business_metrics["prediction_counts"] = prediction_counts
        business_metrics["risk_level_counts"] = risk_level_counts

        return business_metrics

    def display_business_dashboard(
        self, metrics: Dict, business_metrics: Dict, alerts: List[BusinessAlert]
    ):
        """Display business-focused dashboard."""
        print("\n" + "=" * 120)
        print(
            f"🏥 ML BUSINESS METRICS DASHBOARD - "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"⏰ Uptime: " f"{str(datetime.now() - self.start_time).split('.')[0]}")
        print("=" * 120)

        # Model Performance
        print("\n🤖 MODEL PERFORMANCE:")
        print("-" * 60)

        model_accuracy = business_metrics.get("model_accuracy", 0)
        acc_status = (
            "🟢" if model_accuracy >= 0.85 else "🟡" if model_accuracy >= 0.80 else "🔴"
        )
        print(
            f"  Model Accuracy:           {acc_status} "
            f"{model_accuracy:.3f} (Target: >0.80)"
        )

        avg_confidence = business_metrics.get("avg_prediction_confidence", 0)
        conf_status = (
            "🟢" if avg_confidence >= 0.70 else "🟡" if avg_confidence >= 0.60 else "🔴"
        )
        print(
            f"  Avg Prediction Confidence: {conf_status} "
            f"{avg_confidence:.3f} (Target: >0.60)"
        )

        # Business KPIs
        print("\n📊 BUSINESS KPIs:")
        print("-" * 60)

        total_preds = business_metrics.get("total_predictions", 0)
        print(f"  Total Predictions:        📈 {int(total_preds)}")

        pos_rate = business_metrics.get("positive_prediction_rate", 0)
        print(
            f"  Positive Prediction Rate: 🔴 {pos_rate:.3f} " f"({pos_rate*100:.1f}%)"
        )

        neg_rate = business_metrics.get("negative_prediction_rate", 0)
        print(
            f"  Negative Prediction Rate: 🟢 {neg_rate:.3f} " f"({neg_rate*100:.1f}%)"
        )

        # Risk Distribution
        print("\n⚠️  RISK LEVEL DISTRIBUTION:")
        print("-" * 60)

        high_risk = business_metrics.get("high_risk_rate", 0)
        high_status = "🔴" if high_risk > 0.30 else "🟡" if high_risk > 0.20 else "🟢"
        print(
            f"  High Risk:     {high_status} {high_risk:.3f} " f"({high_risk*100:.1f}%)"
        )

        mod_risk = business_metrics.get("moderate_risk_rate", 0)
        print(f"  Moderate Risk: 🟡 {mod_risk:.3f} ({mod_risk*100:.1f}%)")

        low_risk = business_metrics.get("low_risk_rate", 0)
        print(f"  Low Risk:      🟢 {low_risk:.3f} ({low_risk*100:.1f}%)")

        # Prediction Breakdown
        pred_counts = business_metrics.get("prediction_counts", {})
        risk_counts = business_metrics.get("risk_level_counts", {})

        print("\n📋 PREDICTION BREAKDOWN:")
        print("-" * 60)
        print(f"  Positive Predictions: {int(pred_counts.get('positive', 0))}")
        print(f"  Negative Predictions: {int(pred_counts.get('negative', 0))}")
        print(f"  High Risk Cases:      {int(risk_counts.get('high', 0))}")
        print(f"  Moderate Risk Cases:  {int(risk_counts.get('moderate', 0))}")
        print(f"  Low Risk Cases:       {int(risk_counts.get('low', 0))}")

        # Technical Metrics Summary
        print("\n⚡ TECHNICAL PERFORMANCE:")
        print("-" * 60)

        total_requests = sum(metrics.get("request_counts", {}).values())
        error_requests = sum(
            count
            for endpoint, count in metrics.get("request_counts", {}).items()
            if "4xx" in endpoint or "5xx" in endpoint
        )

        if total_requests > 0:
            error_rate = (error_requests / total_requests) * 100
            availability = ((total_requests - error_requests) / total_requests) * 100

            error_status = (
                "🟢" if error_rate <= 1 else "🟡" if error_rate <= 5 else "🔴"
            )
            avail_status = (
                "🟢" if availability >= 99.9 else "🟡" if availability >= 99 else "🔴"
            )

            print(f"  API Availability:  {avail_status} {availability:.2f}%")
            print(f"  Error Rate:        {error_status} {error_rate:.2f}%")
            print(f"  Total Requests:    📊 {int(total_requests)}")

        # Active Business Alerts
        if alerts:
            print(f"\n🚨 BUSINESS ALERTS ({len(alerts)}):")
            print("-" * 60)
            for alert in alerts[-5:]:
                icon = (
                    "🔴"
                    if alert.level == "CRITICAL"
                    else "🟡" if alert.level == "WARNING" else "🔵"
                )
                time_str = alert.timestamp.strftime("%H:%M:%S")
                print(f"  {icon} [{time_str}] {alert.level}: {alert.message}")
                print(f"     💼 Impact: {alert.business_impact}")

        print("\n" + "=" * 120)

    def monitor_business_metrics(self, interval: int = 15):
        """Monitor with business focus."""
        print("🏥 Starting Business Metrics Monitoring...")
        print(f"📡 Fetching metrics every {interval} seconds")
        print("🔔 Business alerting enabled")
        print(
            "💬 Slack alerts: "
            + ("✅ Enabled" if self.slack_alerter.enabled else "❌ Disabled")
        )
        print(
            "📧 Email alerts: "
            + ("✅ Enabled" if self.email_alerter.enabled else "❌ Disabled")
        )
        print("🛑 Press Ctrl+C to stop")

        try:
            while True:
                metrics_text = self.fetch_metrics()
                if metrics_text:
                    # Parse both technical and business metrics
                    from advanced_dashboard import AdvancedMetricsCollector

                    tech_collector = AdvancedMetricsCollector()
                    tech_metrics = tech_collector.parse_metrics(metrics_text)

                    business_metrics = self.parse_business_metrics(metrics_text)

                    # Check for business alerts
                    alerts = self.business_tracker.check_business_alerts(
                        {"business_metrics": business_metrics}
                    )

                    # Display dashboard
                    self.display_business_dashboard(
                        tech_metrics,
                        business_metrics,
                        list(self.business_tracker.alerts),
                    )

                    # Send alerts
                    for alert in alerts:
                        self.slack_alerter.send_alert(alert)
                        # Uncomment and configure for email alerts:
                        # self.email_alerter.send_alert(
                        #     alert, "your-email@example.com",
                        #     "sender@example.com", "password"
                        # )
                else:
                    print(f"❌ Failed to fetch metrics at {datetime.now()}")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Business monitoring stopped by user")

            # Business summary
            print("\n📋 BUSINESS MONITORING SUMMARY:")
            print("-" * 60)
            print(f"Total monitoring time: {datetime.now() - self.start_time}")
            print(f"Total business alerts: {len(self.business_tracker.alerts)}")

            if self.business_tracker.alerts:
                critical_alerts = sum(
                    1
                    for alert in self.business_tracker.alerts
                    if alert.level == "CRITICAL"
                )
                warning_alerts = sum(
                    1
                    for alert in self.business_tracker.alerts
                    if alert.level == "WARNING"
                )
                info_alerts = sum(
                    1 for alert in self.business_tracker.alerts if alert.level == "INFO"
                )
                print(f"Critical alerts: {critical_alerts}")
                print(f"Warning alerts: {warning_alerts}")
                print(f"Info alerts: {info_alerts}")


def main():
    """Main function."""
    collector = BusinessMetricsCollector()

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

    collector.monitor_business_metrics()


if __name__ == "__main__":
    main()
