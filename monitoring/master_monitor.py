#!/usr/bin/env python3
"""
Master Monitoring Script for ML API
Provides a unified interface to all monitoring capabilities.
"""

import sys
import asyncio


def print_banner():
    """Print the monitoring banner."""
    print("=" * 80)
    print("🎛️  ML API COMPREHENSIVE MONITORING SUITE")
    print("=" * 80)
    print("🏥 Heart Disease Prediction API - Production Monitoring")
    print("🚀 Advanced MLOps Observability Platform")
    print("=" * 80)


def print_menu():
    """Print the main menu."""
    print("\n📊 MONITORING OPTIONS:")
    print("-" * 40)
    print("1. 📈 Basic Metrics Dashboard")
    print("2. 🏥 Business Metrics Dashboard")
    print("3. 🎛️  Grafana-Style Dashboard")
    print("4. 🧪 Performance Load Testing")
    print("5. 🔄 Advanced SLA Monitoring")
    print("6. 📧 Configure Alerting")
    print("7. 🐳 Docker Deployment Guide")
    print("8. 📋 Generate Monitoring Report")
    print("9. ❌ Exit")
    print("-" * 40)


def configure_alerting():
    """Configure alerting options."""
    print("\n🔔 ALERTING CONFIGURATION")
    print("=" * 50)

    print("\n💬 SLACK INTEGRATION:")
    print("To enable Slack alerts:")
    print("1. Create a Slack webhook URL")
    print("2. Edit monitoring/business_metrics_dashboard.py")
    print("3. Set webhook_url in SlackAlerter initialization")
    print("   Example: SlackAlerter('https://hooks.slack.com/services/...')")

    print("\n📧 EMAIL INTEGRATION:")
    print("To enable email alerts:")
    print("1. Configure SMTP settings in EmailAlerter")
    print("2. Set enabled=True in email_alerter initialization")
    print("3. Provide email credentials when calling send_alert()")

    print("\n🚨 ALERT THRESHOLDS:")
    print("Current thresholds (edit in BusinessMetricsTracker):")
    print("- Model Accuracy: < 80%")
    print("- Prediction Confidence: < 60%")
    print("- High Risk Rate: > 30%")
    print("- Error Rate: > 5%")
    print("- Latency P95: > 1000ms")

    input("\nPress Enter to continue...")


def docker_deployment_guide():
    """Show Docker deployment guide."""
    print("\n🐳 DOCKER DEPLOYMENT GUIDE")
    print("=" * 50)

    print("\n📋 DEPLOYMENT STEPS:")
    print("1. Ensure Docker Desktop is running")
    print("2. Run: docker-compose up --build")
    print("3. Access services:")
    print("   - API: http://localhost:8000")
    print("   - Prometheus: http://localhost:9090")
    print("   - Grafana: http://localhost:3000")
    print("   - Streamlit: http://localhost:8501")

    print("\n🎛️  GRAFANA SETUP:")
    print("1. Login: admin/admin")
    print("2. Add Prometheus data source: http://prometheus:9090")
    print("3. Import dashboard or create custom panels")

    print("\n📊 PROMETHEUS QUERIES:")
    print("- Request Rate: rate(http_requests_total[1m])")
    print(
        "- P95 Latency: histogram_quantile(0.95, "
        "sum(rate(http_request_duration_seconds_bucket[5m])) by (le))"
    )
    print("- Error Rate: rate(http_requests_total{status_code=~'5..'}[1m])")
    print("- Business Metrics: ml_predictions_total, ml_model_accuracy")

    print("\n🔧 CI/CD INTEGRATION:")
    print("Your docker-compose.yml is ready for:")
    print("- GitHub Actions deployment")
    print("- Kubernetes deployment")
    print("- Cloud platform deployment")

    input("\nPress Enter to continue...")


def generate_monitoring_report():
    """Generate a comprehensive monitoring report."""
    print("\n📋 MONITORING IMPLEMENTATION REPORT")
    print("=" * 60)

    print("\n✅ IMPLEMENTED FEATURES:")
    print("- ✅ Prometheus metrics instrumentation")
    print("- ✅ Custom business metrics tracking")
    print("- ✅ Real-time dashboards (3 types)")
    print("- ✅ Performance load testing")
    print("- ✅ SLA monitoring and alerting")
    print("- ✅ Slack/Email alert integration")
    print("- ✅ Docker-based deployment")
    print("- ✅ Grafana-style visualizations")

    print("\n📊 METRICS TRACKED:")
    print("Technical Metrics:")
    print("- Request counts by endpoint/status")
    print("- Response latency (avg, P95, P99)")
    print("- Error rates and availability")
    print("- System resource usage")

    print("\nBusiness Metrics:")
    print("- Prediction accuracy and confidence")
    print("- Risk level distribution")
    print("- Model performance tracking")
    print("- Feature importance monitoring")

    print("\n🚨 ALERTING CAPABILITIES:")
    print("- SLA threshold monitoring")
    print("- Business metric alerts")
    print("- Multi-channel notifications")
    print("- Configurable thresholds")

    print("\n🎯 PRODUCTION READINESS:")
    print("- ✅ Comprehensive observability")
    print("- ✅ Performance monitoring")
    print("- ✅ Business KPI tracking")
    print("- ✅ Automated alerting")
    print("- ✅ Load testing capabilities")
    print("- ✅ Docker deployment ready")

    print("\n💼 BUSINESS VALUE:")
    print("- Real-time model performance monitoring")
    print("- Early detection of model drift")
    print("- SLA compliance tracking")
    print("- Operational excellence")
    print("- Reduced MTTR (Mean Time To Recovery)")

    input("\nPress Enter to continue...")


async def main():
    """Main function."""
    print_banner()

    while True:
        print_menu()

        try:
            choice = input("\nSelect option (1-9): ").strip()

            if choice == "1":
                print("\n🚀 Starting Basic Metrics Dashboard...")
                import subprocess

                subprocess.run([sys.executable, "monitoring/metrics_collector.py"])

            elif choice == "2":
                print("\n🚀 Starting Business Metrics Dashboard...")
                import subprocess

                subprocess.run(
                    [sys.executable, "monitoring/business_metrics_dashboard.py"]
                )

            elif choice == "3":
                print("\n🚀 Starting Grafana-Style Dashboard...")
                import subprocess

                subprocess.run(
                    [sys.executable, "monitoring/grafana_style_dashboard.py"]
                )

            elif choice == "4":
                print("\n🚀 Starting Performance Load Testing...")
                import subprocess

                subprocess.run([sys.executable, "monitoring/load_tester.py"])

            elif choice == "5":
                print("\n🚀 Starting Advanced SLA Monitoring...")
                import subprocess

                subprocess.run([sys.executable, "monitoring/advanced_dashboard.py"])

            elif choice == "6":
                configure_alerting()

            elif choice == "7":
                docker_deployment_guide()

            elif choice == "8":
                generate_monitoring_report()

            elif choice == "9":
                print("\n👋 Goodbye! Happy monitoring!")
                break

            else:
                print("❌ Invalid choice. Please select 1-9.")

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
