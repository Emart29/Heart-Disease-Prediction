#!/usr/bin/env python3
"""
Performance Testing and Load Monitoring for ML API
Generates realistic load and monitors performance under stress.
"""

import asyncio
import aiohttp
import time
import random
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import statistics
import threading


@dataclass
class LoadTestResult:
    """Results from a load test."""

    timestamp: datetime
    endpoint: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: str = ""


@dataclass
class LoadTestSummary:
    """Summary of load test results."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    duration_seconds: float


class PatientDataGenerator:
    """Generates realistic patient data for testing."""

    def __init__(self):
        self.age_ranges = [(30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]
        self.cp_values = [1.0, 2.0, 3.0, 4.0]
        self.restecg_values = [0.0, 1.0, 2.0]
        self.slope_values = [1.0, 2.0, 3.0]
        self.thal_values = [3.0, 6.0, 7.0]

    def generate_patient(self) -> Dict:
        """Generate a realistic patient data sample."""
        age_range = random.choice(self.age_ranges)
        age = random.randint(age_range[0], age_range[1])

        # Generate correlated values (older patients tend to have higher risk factors)

        return {
            "age": age,
            "sex": random.randint(0, 1),
            "cp": random.choice(self.cp_values),
            "trestbps": random.randint(90, 200),
            "chol": random.randint(120, 400),
            "fbs": random.randint(0, 1),
            "restecg": random.choice(self.restecg_values),
            "thalach": random.randint(60, 200),
            "exang": random.randint(0, 1),
            "oldpeak": round(random.uniform(0, 6), 1),
            "slope": random.choice(self.slope_values),
            "ca": random.randint(0, 4),
            "thal": random.choice(self.thal_values),
        }


class LoadTester:
    """Performs load testing on the ML API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.patient_generator = PatientDataGenerator()
        self.results: List[LoadTestResult] = []
        self.lock = threading.Lock()

    async def make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        method: str = "GET",
        data: Dict = None,
    ) -> LoadTestResult:
        """Make a single request and record the result."""
        start_time = time.time()

        try:
            url = f"{self.base_url}{endpoint}"

            if method == "GET":
                async with session.get(url) as response:
                    await response.text()
                    response_time = (time.time() - start_time) * 1000
                    return LoadTestResult(
                        timestamp=datetime.now(),
                        endpoint=endpoint,
                        status_code=response.status,
                        response_time_ms=response_time,
                        success=200 <= response.status < 300,
                    )
            elif method == "POST":
                async with session.post(url, json=data) as response:
                    await response.text()
                    response_time = (time.time() - start_time) * 1000
                    return LoadTestResult(
                        timestamp=datetime.now(),
                        endpoint=endpoint,
                        status_code=response.status,
                        response_time_ms=response_time,
                        success=200 <= response.status < 300,
                    )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return LoadTestResult(
                timestamp=datetime.now(),
                endpoint=endpoint,
                status_code=0,
                response_time_ms=response_time,
                success=False,
                error_message=str(e),
            )

    async def run_concurrent_requests(
        self,
        num_requests: int,
        concurrent_users: int,
        endpoint: str,
        method: str = "GET",
    ):
        """Run concurrent requests to simulate load."""
        print(
            f"🚀 Starting load test: {num_requests} requests with "
            f"{concurrent_users} concurrent users"
        )
        print(f"🎯 Target endpoint: {method} {endpoint}")

        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            tasks = []

            for i in range(num_requests):
                if endpoint == "/predict":
                    patient_data = self.patient_generator.generate_patient()
                    task = self.make_request(session, endpoint, method, patient_data)
                else:
                    task = self.make_request(session, endpoint, method)

                tasks.append(task)

                # Control concurrency
                if len(tasks) >= concurrent_users:
                    results = await asyncio.gather(*tasks)
                    with self.lock:
                        self.results.extend(results)
                    tasks = []

                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.01)

            # Process remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks)
                with self.lock:
                    self.results.extend(results)

    def analyze_results(self, test_duration: float) -> LoadTestSummary:
        """Analyze load test results."""
        if not self.results:
            return LoadTestSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, test_duration)

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        response_times = [r.response_time_ms for r in successful]

        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[
                18
            ]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[
                98
            ]  # 99th percentile
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = (
                max_response_time
            ) = min_response_time = 0

        return LoadTestSummary(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            success_rate=len(successful) / len(self.results) * 100,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            requests_per_second=(
                len(self.results) / test_duration if test_duration > 0 else 0
            ),
            duration_seconds=test_duration,
        )

    def display_results(self, summary: LoadTestSummary):
        """Display load test results."""
        print("\n" + "=" * 80)
        print("📊 LOAD TEST RESULTS")
        print("=" * 80)

        # Overall Performance
        print("\n🎯 OVERALL PERFORMANCE:")
        print(f"  Total Requests:       {summary.total_requests:,}")
        print(f"  Successful Requests:  {summary.successful_requests:,}")
        print(f"  Failed Requests:      {summary.failed_requests:,}")

        success_icon = (
            "🟢"
            if summary.success_rate >= 99
            else "🟡" if summary.success_rate >= 95 else "🔴"
        )
        print(f"  Success Rate:         {success_icon} {summary.success_rate:.2f}%")

        rps_icon = (
            "🟢"
            if summary.requests_per_second >= 50
            else "🟡" if summary.requests_per_second >= 20 else "🔴"
        )
        print(f"  Requests/Second:      {rps_icon} {summary.requests_per_second:.2f}")

        print(f"  Test Duration:        {summary.duration_seconds:.2f}s")

        # Response Time Analysis
        print("\n⏱️  RESPONSE TIME ANALYSIS:")

        avg_icon = (
            "🟢"
            if summary.avg_response_time <= 200
            else "🟡" if summary.avg_response_time <= 500 else "🔴"
        )
        print(f"  Average Response:     {avg_icon} {summary.avg_response_time:.2f}ms")

        p95_icon = (
            "🟢"
            if summary.p95_response_time <= 500
            else "🟡" if summary.p95_response_time <= 1000 else "🔴"
        )
        print(f"  95th Percentile:      {p95_icon} {summary.p95_response_time:.2f}ms")

        p99_icon = (
            "🟢"
            if summary.p99_response_time <= 1000
            else "🟡" if summary.p99_response_time <= 2000 else "🔴"
        )
        print(f"  99th Percentile:      {p99_icon} {summary.p99_response_time:.2f}ms")

        print(f"  Min Response:         {summary.min_response_time:.2f}ms")
        print(f"  Max Response:         {summary.max_response_time:.2f}ms")

        # Performance Assessment
        print("\n📈 PERFORMANCE ASSESSMENT:")

        if (
            summary.success_rate >= 99
            and summary.avg_response_time <= 200
            and summary.requests_per_second >= 50
        ):
            print("  Overall Rating:       🟢 EXCELLENT - Production ready")
        elif (
            summary.success_rate >= 95
            and summary.avg_response_time <= 500
            and summary.requests_per_second >= 20
        ):
            print("  Overall Rating:       🟡 GOOD - Minor optimizations needed")
        else:
            print(
                "  Overall Rating:       🔴 NEEDS IMPROVEMENT - "
                "Performance issues detected"
            )

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if summary.success_rate < 99:
            print("  - Investigate error causes and improve error handling")
        if summary.avg_response_time > 500:
            print("  - Optimize model inference time")
            print("  - Consider model caching or async processing")
        if summary.requests_per_second < 20:
            print("  - Scale up infrastructure")
            print("  - Implement connection pooling")
        if summary.p95_response_time > 1000:
            print("  - Add request timeout handling")
            print("  - Implement circuit breaker pattern")

        print("\n" + "=" * 80)

    async def run_load_test_suite(self):
        """Run a comprehensive load test suite."""
        print("🧪 COMPREHENSIVE LOAD TEST SUITE")
        print("=" * 50)

        test_scenarios = [
            {
                "name": "Health Check Load",
                "endpoint": "/health",
                "method": "GET",
                "requests": 100,
                "concurrent": 10,
            },
            {
                "name": "Model Info Load",
                "endpoint": "/model-info",
                "method": "GET",
                "requests": 50,
                "concurrent": 5,
            },
            {
                "name": "Light Prediction Load",
                "endpoint": "/predict",
                "method": "POST",
                "requests": 20,
                "concurrent": 2,
            },
            {
                "name": "Medium Prediction Load",
                "endpoint": "/predict",
                "method": "POST",
                "requests": 50,
                "concurrent": 5,
            },
            {
                "name": "Heavy Prediction Load",
                "endpoint": "/predict",
                "method": "POST",
                "requests": 100,
                "concurrent": 10,
            },
        ]

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🧪 Test {i}/{len(test_scenarios)}: {scenario['name']}")
            print("-" * 50)

            self.results = []  # Reset results
            start_time = time.time()

            await self.run_concurrent_requests(
                scenario["requests"],
                scenario["concurrent"],
                scenario["endpoint"],
                scenario["method"],
            )

            test_duration = time.time() - start_time
            summary = self.analyze_results(test_duration)
            self.display_results(summary)

            # Wait between tests
            print("\n⏳ Waiting 5 seconds before next test...")
            await asyncio.sleep(5)

        print("\n🎉 Load test suite completed!")


async def main():
    """Main function to run load tests."""
    print("🔍 Testing API connection...")

    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    print("✅ API is running!")
                else:
                    print(f"⚠️  API returned status code: {response.status}")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure the API is running on http://localhost:8000")
        return

    tester = LoadTester()

    print("\n🎯 Load Test Options:")
    print("1. Quick test (20 requests, 2 concurrent)")
    print("2. Medium test (100 requests, 10 concurrent)")
    print("3. Heavy test (500 requests, 25 concurrent)")
    print("4. Full test suite")

    try:
        choice = input("\nSelect test type (1-4): ").strip()

        if choice == "1":
            print("\n🚀 Running Quick Load Test...")
            await tester.run_concurrent_requests(20, 2, "/predict", "POST")
            summary = tester.analyze_results(10)
            tester.display_results(summary)

        elif choice == "2":
            print("\n🚀 Running Medium Load Test...")
            start_time = time.time()
            await tester.run_concurrent_requests(100, 10, "/predict", "POST")
            test_duration = time.time() - start_time
            summary = tester.analyze_results(test_duration)
            tester.display_results(summary)

        elif choice == "3":
            print("\n🚀 Running Heavy Load Test...")
            start_time = time.time()
            await tester.run_concurrent_requests(500, 25, "/predict", "POST")
            test_duration = time.time() - start_time
            summary = tester.analyze_results(test_duration)
            tester.display_results(summary)

        elif choice == "4":
            await tester.run_load_test_suite()

        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n🛑 Load test interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
