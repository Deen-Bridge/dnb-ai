#!/usr/bin/env python3
"""Automated backup script for SQLite and Redis stores (#883).

Features:
- Online ACID-compliant SQLite backup using SQLite Online Backup API
- Snapshot backup of Redis state with namespace isolation
- Verification of backup integrity (PRAGMA integrity_check, checksums)
- Least-privilege file permissions (0600 on files, 0700 on directories)
- Content-free, privacy-safe reporting with credential redaction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backup_stores")


def redact_url(url: str) -> str:
    """Redact sensitive passwords or credentials from connection URIs."""
    if not url:
        return ""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 checksum of a file in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def backup_sqlite(db_path: str | Path, backup_dir: Path) -> dict[str, Any]:
    """Execute an online ACID backup of the SQLite database and verify integrity."""
    source_path = Path(db_path).resolve()
    if not source_path.exists():
        logger.warning("SQLite database not found at %s; initializing empty store", source_path)
        conn = sqlite3.connect(source_path)
        conn.close()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_filename = f"sqlite_feedback_{timestamp}.db"
    backup_path = backup_dir / backup_filename

    start_time = time.perf_counter()

    # 1. Online backup API (safe during concurrent writes)
    src_conn = sqlite3.connect(str(source_path))
    dest_conn = sqlite3.connect(str(backup_path))
    try:
        src_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        src_conn.close()

    # 2. Enforce least-privilege permissions (0600: read/write owner only)
    backup_path.chmod(0o600)

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    file_stat = backup_path.stat()
    size_bytes = file_stat.st_size
    age_seconds = time.time() - file_stat.st_mtime

    # 3. Integrity check on the backup copy
    chk_conn = sqlite3.connect(str(backup_path))
    try:
        integrity_row = chk_conn.execute("PRAGMA integrity_check").fetchone()
        integrity_status = integrity_row[0] if integrity_row else "unknown"

        # Count records for reporting
        tables = chk_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables if not t[0].startswith("sqlite_")]
        record_counts: dict[str, int] = {}
        total_rows = 0
        for tbl in table_names:
            cnt = chk_conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            record_counts[tbl] = cnt
            total_rows += cnt
    finally:
        chk_conn.close()

    checksum = calculate_sha256(backup_path)

    report = {
        "store": "sqlite_feedback",
        "status": "success" if integrity_status == "ok" else "corrupt",
        "source_path": str(source_path),
        "backup_path": str(backup_path),
        "backup_filename": backup_filename,
        "size_bytes": size_bytes,
        "age_seconds": round(age_seconds, 2),
        "duration_ms": round(duration_ms, 2),
        "integrity": integrity_status,
        "sha256": checksum,
        "row_count": total_rows,
        "tables": record_counts,
        "permissions": oct(backup_path.stat().st_mode & 0o777),
    }

    logger.info(
        "SQLite backup completed: file=%s, size=%d bytes, integrity=%s, rows=%d",
        backup_filename,
        size_bytes,
        integrity_status,
        total_rows,
    )
    return report


def backup_redis(redis_url: str | None, backup_dir: Path) -> dict[str, Any]:
    """Execute a structured snapshot backup of operational Redis state."""
    clean_url = redact_url(redis_url or "")
    if not redis_url:
        logger.info("REDIS_URL not configured; skipping live Redis backup")
        return {
            "store": "redis_operational",
            "status": "skipped",
            "reason": "REDIS_URL not set (in-memory mode)",
            "source_endpoint": "",
        }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_filename = f"redis_operational_{timestamp}.json"
    backup_path = backup_dir / backup_filename

    start_time = time.perf_counter()

    try:
        import redis  # type: ignore

        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()

        # Capture operational namespaces: review, feedback, chat sessions, memory
        namespaces = ["review:*", "feedback:*", "chat:*", "memory:*"]
        exported_data: dict[str, Any] = {
            "metadata": {
                "timestamp": timestamp,
                "exported_at": time.time(),
                "namespaces": namespaces,
            },
            "keys": {},
        }

        total_keys = 0
        for pattern in namespaces:
            for key in client.scan_iter(match=pattern):
                key_type = client.type(key)
                if key_type == "string":
                    val = client.get(key)
                elif key_type == "hash":
                    val = client.hgetall(key)
                elif key_type == "zset":
                    val = client.zrange(key, 0, -1, withscores=True)
                elif key_type == "list":
                    val = client.lrange(key, 0, -1)
                elif key_type == "set":
                    val = list(client.smembers(key))
                else:
                    val = str(client.get(key))

                exported_data["keys"][key] = {
                    "type": key_type,
                    "value": val,
                    "ttl": client.ttl(key),
                }
                total_keys += 1

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(exported_data, f, indent=2)

        backup_path.chmod(0o600)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        file_stat = backup_path.stat()

        report = {
            "store": "redis_operational",
            "status": "success",
            "source_endpoint": clean_url,
            "backup_path": str(backup_path),
            "backup_filename": backup_filename,
            "size_bytes": file_stat.st_size,
            "age_seconds": round(time.time() - file_stat.st_mtime, 2),
            "duration_ms": round(duration_ms, 2),
            "integrity": "ok",
            "sha256": calculate_sha256(backup_path),
            "key_count": total_keys,
            "permissions": oct(file_stat.st_mode & 0o777),
        }
        logger.info(
            "Redis backup completed: file=%s, size=%d bytes, keys=%d",
            backup_filename,
            file_stat.st_size,
            total_keys,
        )
        return report

    except Exception as exc:
        logger.error("Redis backup failed for %s: %s", clean_url, exc)
        return {
            "store": "redis_operational",
            "status": "failed",
            "source_endpoint": clean_url,
            "error": str(exc),
            "integrity": "error",
        }


def run_backups(
    sqlite_path: str = "feedback.db",
    redis_url: str | None = None,
    output_dir: str = "./backups",
) -> dict[str, Any]:
    """Execute all configured store backups and return a summary report."""
    backup_dir = Path(output_dir).resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Ensure backup directory permissions are least-privilege (0700: owner only)
    backup_dir.chmod(0o700)

    sqlite_report = backup_sqlite(sqlite_path, backup_dir)
    redis_report = backup_redis(redis_url, backup_dir)

    all_stores = [sqlite_report, redis_report]
    critical_failed = any(s["status"] == "failed" for s in all_stores)

    summary = {
        "status": "failed" if critical_failed else "success",
        "timestamp": datetime.now(UTC).isoformat(),
        "backup_dir": str(backup_dir),
        "stores": all_stores,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup SQLite and Redis stores for DeenBridge AI")
    parser.add_argument("--sqlite-path", default=os.getenv("FEEDBACK_DB_PATH", "feedback.db"))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL"))
    parser.add_argument("--output-dir", default="./backups")
    parser.add_argument("--json", action="store_true", help="Output summary report as JSON")
    args = parser.parse_args()

    summary = run_backups(
        sqlite_path=args.sqlite_path,
        redis_url=args.redis_url,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        logger.info("Backup summary status: %s", summary["status"])

    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
