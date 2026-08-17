"""Load test for /prediction_api -- verifies the p95 latency < 200ms
requirement. Payloads are self-contained (plausible random values within each
metric's real range) rather than read from dataset.csv, so this runs against
any deployed instance of the service without needing the repo's data files
alongside it.

Usage:
    locust -f locustfile.py --host http://<service-host>
Or headless, with the p95 check enforced as a hard pass/fail:
    locust -f locustfile.py --host http://<service-host> --headless \
        -u 20 -r 5 --run-time 1m
"""
import random

from locust import HttpUser, task, between, events

from prediction_model.config import config

P95_THRESHOLD_MS = 200


def _random_reading():
    return {
        "cpu_usage_pct": round(random.uniform(0, 100), 2),
        "network_in_bytes": round(random.uniform(1_000, 5_000_000), 2),
        "elb_request_count": round(random.uniform(0, 300), 2),
        "rds_cpu_usage_pct": round(random.uniform(0, 100), 2),
    }


def _random_window():
    return {"readings": [_random_reading() for _ in range(config.WINDOW_SIZE)]}


class AnomalyApiUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task
    def predict(self):
        self.client.post("/prediction_api", json=_random_window())


@events.test_stop.add_listener
def check_p95(environment, **kwargs):
    p95 = environment.stats.total.get_response_time_percentile(0.95)
    status = "PASS" if p95 < P95_THRESHOLD_MS else "FAIL"
    print(f"[{status}] p95 response time: {p95:.0f}ms (threshold: {P95_THRESHOLD_MS}ms)")
    if p95 >= P95_THRESHOLD_MS:
        environment.process_exit_code = 1
