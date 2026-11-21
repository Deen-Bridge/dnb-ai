# Backup and Restore

Operational runbook for automated backup verification, restore drills, and disaster recovery
of the DeenBridge AI data stores.

> [!IMPORTANT]
> Never include secrets, passwords, Redis URLs, or database content in log output, CI artifacts,
> or exported reports. All scripts enforce credential redaction via `redact_url()`.

---

## 1. Overview

DeenBridge AI persists operational state across two storage backends:

| Priority | Store | Technology | Contents |
|---|---|---|---|
| Critical | Feedback | SQLite (`feedback.db`) | User thumbs-up/down, categories, comments |
| High | Review queue | Redis (no TTL) | Items pending human review |
| Medium | Sessions | Redis (24 h TTL) | Active conversation context |

This document defines the backup schedule, retention policy, encryption posture, restore
ordering, RPO/RTO targets, and the runbook for periodic drills.

---

## 2. Store Inventory

### 2.1 SQLite Feedback Store

| Property | Value |
|---|---|
| Path | `feedback.db` (default); override with `FEEDBACK_DB_PATH` env var |
| Engine | SQLite (single-file, ACID) |
| Schema | `feedback` table — 11 columns; 3 indexes (rating, created_at, model_name) |
| Backup method | SQLite Online Backup API (`Connection.backup()`) — ACID-consistent hot copy |
| Backup file | `sqlite_feedback_<YYYYMMDDTHHMMSSz>.db` |
| Permissions | 0600 (file), 0700 (directory) |
| Priority | **Critical** — source of truth for all user feedback |

### 2.2 Redis Operational Store

| Property | Value |
|---|---|
| Connection | `REDIS_URL` environment variable (default: `redis://localhost:6379/0`) |
| Contents | Review queue (`review:item:*`, `review:pending`, `review:reviewed`), session keys |
| Backup method | `BGSAVE` snapshot (RDB file) + namespace key export |
| Priority | **High** (review queue) / **Medium** (sessions with 24 h TTL) |
| Notes | Sessions expire automatically; loss requires users to re-authenticate only |

---

## 3. Backup Frequency and Retention

| Store | Frequency | Retention | Method |
|---|---|---|---|
| SQLite feedback | Daily (02:00 UTC via CI) + on-demand | 30 days | Online backup API |
| Redis review queue | Daily (triggered by CI) + on-demand | 30 days | BGSAVE + key export |
| Redis sessions | Daily (best-effort) | 7 days | BGSAVE (sessions self-expire) |

The scheduled CI workflow (`.github/workflows/backup_drill.yml`) runs the backup and
restore drill automatically every day and uploads artifacts retained for 30 days.

On-demand backup:

```bash
python -m scripts.backup_stores \
  --sqlite-path feedback.db \
  --output-dir ./backups \
  --json
```

---

## 4. Encryption at Rest

All backup files are created with **0600 permissions** (owner read/write only) and **0700**
on the containing directory. For transport and long-term storage, encrypt backups before
uploading:

```bash
# Encrypt with GPG symmetric cipher (uses passphrase from env, not shell history)
gpg --batch --yes --passphrase-file <(echo "$BACKUP_PASSPHRASE") \
    --symmetric --cipher-algo AES256 \
    -o backups/sqlite_feedback.db.gpg \
    backups/sqlite_feedback_*.db

# Decrypt for restore
gpg --batch --yes --passphrase-file <(echo "$BACKUP_PASSPHRASE") \
    --decrypt backups/sqlite_feedback.db.gpg > restored_feedback.db
```

For cloud-managed environments, use the provider KMS (e.g., Google Cloud KMS,
AWS KMS) to encrypt the backup bucket rather than local GPG.

> [!CAUTION]
> Never store `BACKUP_PASSPHRASE` in code, CI environment variables visible in logs,
> or unencrypted `.env` files. Use a secrets manager (e.g., GitHub Secrets, Vault).

---

## 5. Restore Order

Restore stores in this order to preserve referential consistency:

1. **SQLite feedback** (`feedback.db`) — stateless; restore first because review queue
   items reference feedback IDs.
2. **Redis sessions** — restore before review queue so session tokens are valid when
   the review workflow resumes.
3. **Redis review queue** — restore last; items may reference session and feedback state
   restored in steps 1–2.

---

## 6. RPO / RTO Targets

| Target | Value | Measurement |
|---|---|---|
| Recovery Point Objective (RPO) | **≤ 1 hour** (3600 s) | Age of the most recent backup at time of incident |
| Recovery Time Objective (RTO) | **≤ 15 minutes** (900 s) | Time from restore start to verified recovery |

The restore drill script measures both and fails the drill if either target is exceeded.
CI enforces these thresholds on every scheduled run.

---

## 7. Running the Backup Script

```bash
# Minimal — SQLite only
python -m scripts.backup_stores \
  --sqlite-path feedback.db \
  --output-dir ./backups

# With Redis (REDIS_URL must be set)
REDIS_URL=redis://localhost:6379/0 \
  python -m scripts.backup_stores \
  --sqlite-path feedback.db \
  --output-dir ./backups \
  --json
```

The script outputs a JSON report (one object per store) with:

```json
{
  "store": "sqlite_feedback",
  "status": "success",
  "backup_path": "backups/sqlite_feedback_20260301T020000Z.db",
  "size_bytes": 102400,
  "age_seconds": 0.12,
  "sha256": "a3f9...c1e2",
  "integrity": "ok",
  "row_count": 1842,
  "duration_seconds": 0.085,
  "timestamp": "2026-03-01T02:00:00Z"
}
```

> [!NOTE]
> `backup_stores.py` **never logs prompts, answers, or user content**. The report
> contains only structural metadata (counts, sizes, hashes, durations).

---

## 8. Running the Restore Drill

```bash
# Run against existing backups in ./backups/
python -m scripts.restore_drill \
  --backup-dir ./backups \
  --json \
  --output-report ./backups/restore-report.json

# Generate fresh backup then drill
python -m scripts.restore_drill \
  --backup-dir ./backups \
  --run-backup-first \
  --sqlite-path feedback.db \
  --json \
  --output-report ./backups/restore-report.json
```

The drill:
1. Copies each backup into an isolated `tempfile.TemporaryDirectory` (deleted on exit).
2. Runs `PRAGMA integrity_check` and `PRAGMA foreign_key_check`.
3. Verifies every required column and index exists.
4. Samples up to 50 records for type and JSON validity.
5. Measures RTO and RPO against the targets in §6.
6. Writes a machine-readable JSON report and exits non-zero on any failure.

**Sample passing report:**

```json
{
  "status": "passed",
  "timestamp": "2026-03-01T02:01:15Z",
  "total_duration_seconds": 0.34,
  "failures": [],
  "stores_verified": [
    {
      "store": "sqlite_feedback",
      "status": "passed",
      "integrity_check": "ok",
      "schema_verified": true,
      "columns_verified": 11,
      "indexes_verified": 3,
      "records_sampled": 50,
      "total_records": 1842,
      "rto_seconds": 0.21,
      "rpo_seconds": 72.4,
      "rto_target_met": true,
      "rpo_target_met": true
    }
  ]
}
```

---

## 9. Automated CI Drill

The workflow `.github/workflows/backup_drill.yml` runs every day at **02:00 UTC** and
on `workflow_dispatch`. It:

1. Checks out the repository.
2. Installs `requirements-dev.txt`.
3. Creates `backups/` with mode `0700`.
4. Runs `backup_stores.py` (never logs `REDIS_URL` or passwords).
5. Runs `restore_drill.py` and writes `backups/restore-report.json`.
6. Verifies the report status (exits non-zero on failure).
7. Uploads `backups/` as a GitHub Actions artifact (retained 30 days).
8. On failure, prints an actionable alert with escalation steps and links to this runbook.

Configure `REDIS_URL` as a repository secret to include Redis in the drill. Without it,
only the SQLite drill runs.

---

## 10. Staging Verification Checklist

Before every production deployment verify:

- [ ] **Non-root execution**: `docker run --rm <image> id` outputs `uid=1000(app)` (not root)
- [ ] **Required write paths**: `/tmp` is writable; `feedback.db` path is writable by the app user
- [ ] **Backup completes**: `python -m scripts.backup_stores --sqlite-path feedback.db --output-dir /tmp/stage-backup --json` exits 0
- [ ] **Restore drill passes**: `python -m scripts.restore_drill --backup-dir /tmp/stage-backup --json` exits 0
- [ ] **RPO/RTO met**: `rto_target_met` and `rpo_target_met` are both `true` in the report

---

## 11. Rollback to a Known-Good Backup

To restore to a specific point in time:

```bash
# 1. List available backups
ls -lh backups/sqlite_feedback_*.db

# 2. Identify the target backup by timestamp
TARGET=backups/sqlite_feedback_20260228T020000Z.db

# 3. Verify the backup before restoring (run the drill against it)
python -m scripts.restore_drill --backup-dir backups --json

# 4. Stop the application (prevent new writes during restore)
# (Use your process manager: systemctl stop dnb-ai / docker stop <container>)

# 5. Replace the live database (keep a copy of the bad one for investigation)
cp feedback.db feedback.db.pre-rollback
cp "$TARGET" feedback.db
chmod 600 feedback.db

# 6. Restart the application
# (systemctl start dnb-ai / docker start <container>)

# 7. Verify the application is healthy
curl -f http://localhost:8000/health
```

---

## 12. Release Evidence

Attach the following to every production release that touches data persistence:

| Artifact | Source |
|---|---|
| Backup inventory | Output of `backup_stores.py --json` |
| Latest restore report | `backups/restore-report.json` from CI |
| RPO/RTO measurements | `rto_seconds` and `rpo_seconds` in the restore report |
| Drill CI link | Link to the latest passing `Backup and Restore Drill` workflow run |

---

## 13. Runbook Owner

**Primary**: Platform team on-call  
**Escalation**: Engineering lead / repository admin  
**Review cadence**: Quarterly drill review and RPO/RTO target reassessment

For incidents: open an issue tagged `[PROD-INCIDENT]` and page the platform on-call
via the standard escalation path.
