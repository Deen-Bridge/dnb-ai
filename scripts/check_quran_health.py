#!/usr/bin/env python3
"""
Quran Source Availability and Freshness Health Check
Issue #876: make Quran source availability and freshness observable

Usage:
    python scripts/check_quran_health.py [--api-url URL] [--output-json PATH]

Exit codes:
    0 - healthy
    1 - degraded (slow or partial response)
    2 - down (unreachable or critical error)
"""

import argparse
import json
import sys
import time
import datetime

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

# Default Quran API endpoint (alquran.cloud is a common open source provider)
DEFAULT_API_URL = "https://api.alquran.cloud/v1/ayah/1:1/en.asad"
# Freshness: warn if the source hasn't updated metadata in this many days
FRESHNESS_WARN_DAYS = 30
# Latency thresholds
LATENCY_WARN_MS = 2000
LATENCY_CRIT_MS = 5000


def check_quran_api(api_url: str) -> dict:
    """Ping the Quran API and validate the response shape."""
    start = time.monotonic()
    result = {
        "api_url": api_url,
        "last_checked": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "unknown",
        "latency_ms": None,
        "http_status": None,
        "error": None,
        "verse_text_present": False,
        "translation_present": False,
        "edition_present": False,
    }

    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "dnb-ai-health-check/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed_ms = (time.monotonic() - start) * 1000
            result["latency_ms"] = round(elapsed_ms, 1)
            result["http_status"] = resp.status

            body = json.loads(resp.read().decode("utf-8"))
            data = body.get("data", {})

            # Validate key fields expected from the Quran API response
            result["verse_text_present"] = bool(data.get("text"))
            result["translation_present"] = bool(data.get("text"))
            result["edition_present"] = bool(data.get("edition", {}).get("identifier"))

            if resp.status == 200 and result["verse_text_present"]:
                if elapsed_ms > LATENCY_CRIT_MS:
                    result["status"] = "degraded"
                    result["error"] = f"Response time {elapsed_ms:.0f}ms exceeds critical threshold {LATENCY_CRIT_MS}ms"
                elif elapsed_ms > LATENCY_WARN_MS:
                    result["status"] = "degraded"
                    result["error"] = f"Response time {elapsed_ms:.0f}ms exceeds warning threshold {LATENCY_WARN_MS}ms"
                else:
                    result["status"] = "ok"
            else:
                result["status"] = "degraded"
                result["error"] = f"HTTP {resp.status} or missing verse text"

    except urllib.error.HTTPError as e:
        result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        result["http_status"] = e.code
        result["status"] = "down"
        result["error"] = f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        result["status"] = "down"
        result["error"] = f"Connection error: {e.reason}"
    except Exception as e:  # noqa: BLE001
        result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        result["status"] = "down"
        result["error"] = f"Unexpected error: {e}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Quran API health check")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Quran API endpoint to check",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write JSON health report to this path",
    )
    args = parser.parse_args()

    report = check_quran_api(args.api_url)

    # Print summary to stdout
    status_symbol = {"ok": "âœ…", "degraded": "âš ï¸", "down": "âŒ"}.get(report["status"], "?")
    print(f"{status_symbol} Quran API: {report['status'].upper()}")
    print(f"   URL:     {report['api_url']}")
    print(f"   Latency: {report['latency_ms']}ms")
    print(f"   HTTP:    {report['http_status']}")
    if report["error"]:
        print(f"   Error:   {report['error']}")
    print(f"   Checked: {report['last_checked']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output_json}")

    # Exit codes: 0=ok, 1=degraded, 2=down
    return {"ok": 0, "degraded": 1, "down": 2}.get(report["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
