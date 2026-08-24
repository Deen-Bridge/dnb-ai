# Performance baseline

The performance suite runs Locust against the service with `MOCK_UPSTREAMS=1`. Mock mode replaces Gemini, Horizon, and nisab price retrieval with deterministic local seams, so these results measure DeenBridge request handling rather than external provider latency or quota.

Run locally:

```bash
MOCK_UPSTREAMS=1 SAFETY_PIPELINE_ENABLED=false AUTH_DISABLED=true uvicorn main:app --port 8000
locust -f loadtest/locustfile.py --headless -H http://127.0.0.1:8000 -u 10 -r 2 -t 45s --csv=/tmp/locust
python loadtest/check_budget.py loadtest/budget.yaml /tmp/locust_stats.csv
```

The CI budget intentionally uses broad thresholds because hosted runners are noisy. Update `loadtest/budget.yaml` only when a deliberate performance change is reviewed; include the old/new p95, throughput, and error-rate measurements in the pull request.

| Endpoint/profile | Concurrency | p50 | p95 | p99 | RPS | Error rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `/chat` single-turn | 1/10/50 | CI artifact | CI artifact | CI artifact | CI artifact | CI artifact |
| `/chat` multi-turn | 1/10/50 | CI artifact | CI artifact | CI artifact | CI artifact | CI artifact |
| `/zakat` success/errors | 1/10/50 | CI artifact | CI artifact | CI artifact | CI artifact | CI artifact |

Initial measurements should be used to quantify the existing blocking-upstream and unbounded-history issues; this issue does not change those algorithms.
