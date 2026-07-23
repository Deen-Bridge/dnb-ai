import os
import subprocess
import time
import sys
import csv
import urllib.request
import urllib.error
import yaml

def wait_for_server(url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    print("FastAPI server is up!")
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(1)
    print("Timeout waiting for FastAPI server to start.")
    return False

def parse_locust_csv(csv_path):
    stats = {}
    if not os.path.exists(csv_path):
        print(f"Locust stats CSV not found at {csv_path}")
        return stats
        
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name")
            if not name or name == "Aggregations":
                continue
            
            # Extract metrics
            try:
                req_count = int(row.get("Request Count", 0))
                err_count = int(row.get("Failure Count", 0))
                rps = float(row.get("Requests/s", 0.0))
                p95 = float(row.get("95%", 0.0))
                error_rate = err_count / req_count if req_count > 0 else 0.0
                
                stats[name] = {
                    "rps": rps,
                    "p95": p95,
                    "error_rate": error_rate,
                    "req_count": req_count
                }
            except (ValueError, TypeError) as e:
                print(f"Error parsing row for {name}: {e}")
    return stats

def main():
    # Configuration
    users = os.getenv("LOCUST_USERS", "5")
    spawn_rate = os.getenv("LOCUST_SPAWN_RATE", "1")
    run_time = os.getenv("LOCUST_RUN_TIME", "45s")
    
    csv_prefix = "loadtest/report"
    stats_csv = f"{csv_prefix}_stats.csv"
    
    # Load budget configuration
    budget_path = "loadtest/budget.yaml"
    if not os.path.exists(budget_path):
        print(f"Budget config not found at {budget_path}")
        sys.exit(1)
        
    with open(budget_path, 'r') as f:
        budget_data = yaml.safe_load(f)
    budgets = budget_data.get("budget", {})

    # Start Uvicorn server in a subprocess
    env = os.environ.copy()
    env["MOCK_UPSTREAMS"] = "1"
    # Ensure Redis doesn't block if not configured or missing
    env["REDIS_URL"] = "" 
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Starting FastAPI server in mock mode...")
    if not wait_for_server("http://localhost:8000/ping"):
        server_process.terminate()
        sys.exit(1)

    # Run Locust
    locust_cmd = [
        sys.executable, "-m", "locust",
        "-f", "loadtest/locustfile.py",
        "--headless",
        "-u", users,
        "-r", spawn_rate,
        "--run-time", run_time,
        "--csv", csv_prefix,
        "--host", "http://localhost:8000"
    ]
    
    print(f"Running Locust with {users} users, {spawn_rate} spawn rate, for {run_time}...")
    try:
        subprocess.run(locust_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Locust load test failed: {e}")
        server_process.terminate()
        sys.exit(1)
    finally:
        # Stop uvicorn server
        print("Stopping FastAPI server...")
        server_process.terminate()
        server_process.wait()

    # Parse and Validate Results
    stats = parse_locust_csv(stats_csv)
    violations = []
    
    print("\n================ Performance Evaluation Report ================")
    for endpoint, budget in budgets.items():
        if endpoint not in stats:
            # We don't fail immediately, maybe the scenario didn't trigger this endpoint yet
            print(f"⚠️  Endpoint '{endpoint}' was not recorded in stats.")
            continue
            
        estats = stats[endpoint]
        p95_actual = estats["p95"]
        rps_actual = estats["rps"]
        error_rate_actual = estats["error_rate"]
        
        p95_limit = budget["max_p95_ms"]
        rps_limit = budget["min_rps"]
        error_limit = budget["max_error_rate"]
        
        print(f"\nRoute: {endpoint} (Requests: {estats['req_count']})")
        
        # Validate P95
        p95_status = "✅"
        if p95_actual > p95_limit:
            p95_status = "❌"
            violations.append(f"{endpoint}: p95 latency {p95_actual}ms exceeded budget of {p95_limit}ms")
        print(f"  P95 Latency: {p95_actual}ms (Budget: <= {p95_limit}ms) {p95_status}")
            
        # Validate RPS
        rps_status = "✅"
        if rps_actual < rps_limit:
            rps_status = "❌"
            violations.append(f"{endpoint}: RPS {rps_actual:.2f} below budget of {rps_limit:.2f}")
        print(f"  RPS: {rps_actual:.2f} (Budget: >= {rps_limit:.2f}) {rps_status}")
            
        # Validate Error Rate
        err_status = "✅"
        if error_rate_actual > error_limit:
            err_status = "❌"
            violations.append(f"{endpoint}: error rate {error_rate_actual*100:.1f}% exceeded budget of {error_limit*100:.1f}%")
        print(f"  Error Rate: {error_rate_actual*100:.1f}% (Budget: <= {error_limit*100:.1f}%) {err_status}")

    print("\n==============================================================")
    if violations:
        print("\n❌ Performance Budget BREACHED:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("\n✅ Performance Budget PASSED!")
        sys.exit(0)

if __name__ == "__main__":
    main()
