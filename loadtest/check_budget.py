import csv
import sys

import yaml


def main() -> int:
    budget = yaml.safe_load(open(sys.argv[1]))
    rows = {row["Name"]: row for row in csv.DictReader(open(sys.argv[2]))}
    failures = []
    for name, limits in budget.items():
        row = rows.get(name)
        if not row:
            failures.append(f"{name}: no CSV row")
            continue
        p95 = float(row["95%"])
        rps = float(row["Requests/s"])
        errors = int(row["Failure Count"])
        requests = int(row["Request Count"])
        error_rate = errors / requests if requests else 1.0
        if p95 > limits["max_p95_ms"]:
            failures.append(f"{name}: p95 {p95:.1f}ms > {limits['max_p95_ms']}ms")
        if rps < limits["min_rps"]:
            failures.append(f"{name}: rps {rps:.3f} < {limits['min_rps']}")
        if error_rate > limits["max_error_rate"]:
            failures.append(f"{name}: error rate {error_rate:.3f} > {limits['max_error_rate']}")
    if failures:
        print("Performance budget failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Performance budget passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
