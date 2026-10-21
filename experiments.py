"""Durable, process-safe experiment configuration and rollback control.

Reuse the prompt registry's sticky assignment rather than creating a second
A/B router. SQLite transactions coordinate workers sharing the same database;
no prompts, session identifiers, or generated answers enter the metrics store.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prompts.registry import ExperimentConfig, ExperimentHarness, PromptRegistry, Variant
from prompts.template import PromptTemplate

logger = logging.getLogger(__name__)

RUNTIME_VARIABLES = frozenset(
    {"response_language", "language", "madhhab", "knowledge_level", "intent", "retrieved_context", "user_question"}
)


class GenerationOverrides(BaseModel):
    """Only generation settings supported by the chat handler may vary."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    max_output_tokens: int | None = Field(None, ge=1)


class RollbackPolicy(BaseModel):
    """Evaluate treatment error rate and nearest-rank p95 over a bounded window."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min_samples: int = Field(20, ge=1, le=10000)
    window_size: int = Field(100, ge=1, le=10000)
    max_error_rate: float | None = Field(0.1, ge=0, le=1)
    max_p95_latency_ms: float | None = Field(5000, gt=0)

    @model_validator(mode="after")
    def validate_window(self) -> RollbackPolicy:
        if self.min_samples > self.window_size:
            raise ValueError("min_samples must not exceed window_size")
        return self


@dataclass(frozen=True)
class ExperimentDecision:
    experiment_id: str
    version: int
    variant: Variant
    template: PromptTemplate
    is_control: bool
    rolled_back: bool = False
    stable: ExperimentDecision | None = None


@dataclass
class _Version:
    config: ExperimentConfig
    policy: RollbackPolicy
    templates: dict[str, PromptTemplate]


class ExperimentManager:
    """Versioned configuration, feature switches, metrics, and audited rollback.

    The first registered control is the stable baseline. Later registrations
    preserve it until an administrator explicitly marks another version stable.
    Configuration and prompt bodies are snapshotted, so updating a registry or
    restarting a worker cannot silently change the rollback target.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        default_model: str,
        database: str = ":memory:",
        default_generation_config: dict[str, float | int] | None = None,
    ) -> None:
        self._registry = registry
        self._default_model = default_model
        self._default_generation_config = dict(default_generation_config or {})
        self._lock = RLock()
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(database, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS versions (
                experiment TEXT, version INTEGER, payload TEXT NOT NULL,
                PRIMARY KEY (experiment, version)
            );
            CREATE TABLE IF NOT EXISTS states (
                experiment TEXT PRIMARY KEY, current_version INTEGER NOT NULL,
                stable_version INTEGER NOT NULL, killed INTEGER NOT NULL, rollback_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS runtime (id INTEGER PRIMARY KEY, experiment TEXT);
            INSERT OR IGNORE INTO runtime VALUES (1, NULL);
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY, experiment TEXT, version INTEGER, variant TEXT,
                latency_ms REAL NOT NULL, error INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS observations_bucket ON observations (experiment, version, variant, id);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, experiment TEXT, event TEXT, version INTEGER,
                previous_version INTEGER, reason TEXT, timestamp TEXT
            );
            """
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    def _state(self, db: sqlite3.Connection, experiment_id: str) -> sqlite3.Row:
        state = db.execute("SELECT * FROM states WHERE experiment = ?", (experiment_id,)).fetchone()
        if state is None:
            raise KeyError(experiment_id)
        return state

    def _version(self, db: sqlite3.Connection, experiment_id: str, version: int) -> _Version:
        row = db.execute(
            "SELECT payload FROM versions WHERE experiment = ? AND version = ?", (experiment_id, version)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown configuration version: {version}")
        data = json.loads(row["payload"])
        config = data["config"]
        config["control"] = Variant(**config["control"])
        config["variants"] = [Variant(**v) for v in config["variants"]]
        templates = {}
        for name, template in data["templates"].items():
            template["variables"] = tuple(template["variables"])
            templates[name] = PromptTemplate(**template)
        return _Version(ExperimentConfig(**config), RollbackPolicy(**data["policy"]), templates)

    def _event(
        self,
        db: sqlite3.Connection,
        experiment_id: str,
        event: str,
        version: int,
        previous: int | None = None,
        reason: str | None = None,
    ) -> None:
        db.execute(
            "INSERT INTO events (experiment,event,version,previous_version,reason,timestamp) VALUES (?,?,?,?,?,?)",
            (experiment_id, event, version, previous, reason, datetime.now(UTC).isoformat()),
        )
        logger.info(
            "experiment state changed",
            extra={"experiment_id": experiment_id, "event": event, "version": version, "reason": reason},
        )

    def register_experiment(self, config: ExperimentConfig, policy: RollbackPolicy | None = None) -> int:
        """Validate and stage a revision; updating an experiment never activates it."""
        pinned = copy.deepcopy(config)
        if not pinned.experiment_id or len(pinned.experiment_id) > 128:
            raise ValueError("experiment_id must contain 1 to 128 characters")
        if pinned.traffic_percent is not None and (
            not math.isfinite(pinned.traffic_percent) or not 0 <= pinned.traffic_percent <= 100
        ):
            raise ValueError("traffic_percent must be between 0 and 100")
        templates = {}
        variants = [pinned.control, *pinned.variants]
        if len({v.name for v in variants}) != len(variants):
            raise ValueError("Variant names must be unique")
        if not math.isfinite(sum(v.weight for v in variants)):
            raise ValueError("Total variant weight must be finite")
        for variant in variants:
            if not variant.name or not math.isfinite(variant.weight) or variant.weight <= 0:
                raise ValueError("Variant names must be nonempty and weights finite and positive")
            template = self._registry.get(variant.template_name, variant.template_version)
            if template is None:
                raise ValueError(f"Unknown template: {variant.template_name}")
            if set(template.variables) - RUNTIME_VARIABLES:
                raise ValueError("Template variables are not supported by chat experiments")
            variant.template_version = template.version
            if variant.model_name is None:
                variant.model_name = self._default_model
            if not variant.model_name.strip() or len(variant.model_name) > 128:
                raise ValueError("model_name must contain 1 to 128 characters")
            overrides = GenerationOverrides(**{**self._default_generation_config, **variant.generation_config})
            variant.generation_config = overrides.model_dump(exclude_none=True)
            templates[variant.name] = asdict(template)
        payload = json.dumps(
            {"config": asdict(pinned), "policy": (policy or RollbackPolicy()).model_dump(), "templates": templates},
            allow_nan=False,
        )
        with self._transaction() as db:
            previous = db.execute("SELECT * FROM states WHERE experiment = ?", (pinned.experiment_id,)).fetchone()
            version = db.execute(
                "SELECT COALESCE(MAX(version),0) + 1 FROM versions WHERE experiment = ?", (pinned.experiment_id,)
            ).fetchone()[0]
            db.execute("INSERT INTO versions VALUES (?,?,?)", (pinned.experiment_id, version, payload))
            if previous is None:
                db.execute(
                    "INSERT INTO states VALUES (?,?,?,?,NULL)",
                    (pinned.experiment_id, version, version, pinned.kill_switch),
                )
            self._event(
                db, pinned.experiment_id, "registered", version, previous["current_version"] if previous else None
            )
        return int(version)

    def activate(self, experiment_id: str) -> None:
        with self._transaction() as db:
            state = self._state(db, experiment_id)
            version = int(
                db.execute("SELECT MAX(version) FROM versions WHERE experiment=?", (experiment_id,)).fetchone()[0]
            )
            if state["rollback_reason"] is not None:
                rejected = db.execute(
                    "SELECT MAX(previous_version) FROM events WHERE experiment=? AND event='rolled_back'",
                    (experiment_id,),
                ).fetchone()[0]
                if rejected is not None and version <= rejected:
                    raise ValueError(
                        "Register a new configuration version before reactivating a rolled-back experiment"
                    )
            data = self._version(db, experiment_id, version)
            previous = db.execute("SELECT experiment FROM runtime WHERE id=1").fetchone()[0]
            if previous is not None and previous != experiment_id:
                old = self._state(db, previous)
                self._event(db, previous, "deactivated", old["current_version"])
            db.execute("UPDATE runtime SET experiment=? WHERE id=1", (experiment_id,))
            db.execute(
                "UPDATE states SET current_version=?,killed=?,rollback_reason=NULL WHERE experiment=?",
                (version, data.config.kill_switch, experiment_id),
            )
            self._event(db, experiment_id, "activated", version, state["current_version"])

    def deactivate(self, experiment_id: str) -> None:
        with self._transaction() as db:
            state = self._state(db, experiment_id)
            db.execute("UPDATE runtime SET experiment=NULL WHERE id=1 AND experiment=?", (experiment_id,))
            self._event(db, experiment_id, "deactivated", state["current_version"])

    def _decision(
        self, experiment_id: str, version: int, data: _Version, name: str, rolled_back: bool
    ) -> ExperimentDecision:
        variant = next(v for v in [data.config.control, *data.config.variants] if v.name == name)
        return ExperimentDecision(
            experiment_id, version, variant, data.templates[name], name == data.config.control.name, rolled_back
        )

    def choose(self, session_id: str) -> ExperimentDecision | None:
        """Choose from the current shared snapshot; killed experiments use stable control."""
        with self._transaction() as db:
            experiment_id = db.execute("SELECT experiment FROM runtime WHERE id=1").fetchone()[0]
            if experiment_id is None:
                return None
            state = self._state(db, experiment_id)
            version = state["stable_version"] if state["killed"] else state["current_version"]
            data = self._version(db, experiment_id, version)
            if state["killed"]:
                return self._decision(experiment_id, version, data, data.config.control.name, True)
            harness = ExperimentHarness(self._registry)
            harness.register_experiment(replace(data.config, kill_switch=False))
            assignment = harness.assign(experiment_id, session_id)
            stable = self._version(db, experiment_id, state["stable_version"])
            stable_decision = self._decision(
                experiment_id, state["stable_version"], stable, stable.config.control.name, True
            )
            if assignment.variant_name == data.config.control.name:
                # A staged control change is not trusted until explicitly
                # marked stable; untreated traffic always uses known stable.
                return replace(stable_decision, rolled_back=False)
            return replace(
                self._decision(experiment_id, version, data, assignment.variant_name, False),
                stable=stable_decision,
            )

    def stable_decision(self, experiment_id: str) -> ExperimentDecision:
        with self._transaction() as db:
            state = self._state(db, experiment_id)
            data = self._version(db, experiment_id, state["stable_version"])
            return self._decision(experiment_id, state["stable_version"], data, data.config.control.name, True)

    def set_killed(self, experiment_id: str, killed: bool) -> None:
        with self._transaction() as db:
            state = self._state(db, experiment_id)
            if not killed and state["rollback_reason"] is not None:
                raise ValueError(
                    "Register a new configuration version before resuming an automatically or manually rolled-back experiment"
                )
            db.execute("UPDATE states SET killed=? WHERE experiment=?", (killed, experiment_id))
            self._event(db, experiment_id, "killed" if killed else "resumed", state["current_version"])

    def mark_stable(self, experiment_id: str, version: int) -> None:
        with self._transaction() as db:
            state = self._state(db, experiment_id)
            self._version(db, experiment_id, version)
            db.execute("UPDATE states SET stable_version=? WHERE experiment=?", (version, experiment_id))
            self._event(db, experiment_id, "marked_stable", version, state["stable_version"])

    def _rollback(self, db: sqlite3.Connection, experiment_id: str, reason: str) -> None:
        state = self._state(db, experiment_id)
        db.execute(
            "UPDATE states SET current_version=stable_version,killed=1,rollback_reason=? WHERE experiment=?",
            (reason, experiment_id),
        )
        self._event(db, experiment_id, "rolled_back", state["stable_version"], state["current_version"], reason)

    def rollback(self, experiment_id: str) -> None:
        with self._transaction() as db:
            self._rollback(db, experiment_id, "manual")

    def record(self, decision: ExperimentDecision, latency_ms: float, error: bool) -> bool:
        """Record an actual model attempt and roll back breached treatment windows once."""
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency_ms must be finite and nonnegative")
        with self._transaction() as db:
            state = db.execute("SELECT * FROM states WHERE experiment=?", (decision.experiment_id,)).fetchone()
            if state is None:
                return False
            policy = self._version(db, decision.experiment_id, decision.version).policy
            bucket = (decision.experiment_id, decision.version, decision.variant.name)
            db.execute(
                "INSERT INTO observations (experiment,version,variant,latency_ms,error) VALUES (?,?,?,?,?)",
                (*bucket, latency_ms, error),
            )
            db.execute(
                "DELETE FROM observations WHERE experiment=? AND version=? AND variant=? AND id NOT IN "
                "(SELECT id FROM observations WHERE experiment=? AND version=? AND variant=? ORDER BY id DESC LIMIT ?)",
                (*bucket, *bucket, policy.window_size),
            )
            samples = db.execute(
                "SELECT latency_ms,error FROM observations WHERE experiment=? AND version=? AND variant=?", bucket
            ).fetchall()
            if (
                decision.is_control
                or state["killed"]
                or state["current_version"] != decision.version
                or len(samples) < policy.min_samples
            ):
                return False
            error_rate = sum(row["error"] for row in samples) / len(samples)
            p95 = sorted(row["latency_ms"] for row in samples)[math.ceil(len(samples) * 0.95) - 1]
            reason = None
            if policy.max_error_rate is not None and error_rate > policy.max_error_rate:
                reason = "error_rate"
            elif policy.max_p95_latency_ms is not None and p95 > policy.max_p95_latency_ms:
                reason = "p95_latency"
            if reason:
                self._rollback(db, decision.experiment_id, reason)
                return True
            return False

    def list_experiments(self) -> dict[str, object]:
        with self._transaction() as db:
            result: dict[str, object] = {}
            active = db.execute("SELECT experiment FROM runtime WHERE id=1").fetchone()[0]
            for state in db.execute("SELECT * FROM states ORDER BY experiment").fetchall():
                data = self._version(db, state["experiment"], state["current_version"])
                result[state["experiment"]] = {
                    **asdict(data.config),
                    "variants": [asdict(data.config.control), *[asdict(v) for v in data.config.variants]],
                    "kill_switch": bool(state["killed"]),
                    "version": state["current_version"],
                    "latest_version": db.execute(
                        "SELECT MAX(version) FROM versions WHERE experiment=?", (state["experiment"],)
                    ).fetchone()[0],
                    "stable_version": state["stable_version"],
                    "active": active == state["experiment"],
                    "rollback_reason": state["rollback_reason"],
                    "rollback_policy": data.policy.model_dump(),
                }
            return result

    def versions(self, experiment_id: str) -> list[dict[str, object]]:
        with self._transaction() as db:
            self._state(db, experiment_id)
            return [
                {"version": row["version"], **json.loads(row["payload"])}
                for row in db.execute("SELECT * FROM versions WHERE experiment=? ORDER BY version", (experiment_id,))
            ]

    def metrics(self, experiment_id: str) -> list[dict[str, object]]:
        with self._transaction() as db:
            self._state(db, experiment_id)
            buckets = db.execute(
                "SELECT DISTINCT version,variant FROM observations WHERE experiment=? ORDER BY version,variant",
                (experiment_id,),
            ).fetchall()
            result = []
            for bucket in buckets:
                rows = db.execute(
                    "SELECT latency_ms,error FROM observations WHERE experiment=? AND version=? AND variant=?",
                    (experiment_id, bucket["version"], bucket["variant"]),
                ).fetchall()
                result.append(
                    {
                        "version": bucket["version"],
                        "variant": bucket["variant"],
                        "window_attempts": len(rows),
                        "errors": sum(row["error"] for row in rows),
                        "error_rate": sum(row["error"] for row in rows) / len(rows),
                        "p95_latency_ms": sorted(row["latency_ms"] for row in rows)[math.ceil(len(rows) * 0.95) - 1],
                    }
                )
            return result

    def events(self, experiment_id: str) -> list[dict[str, object]]:
        with self._transaction() as db:
            return [
                dict(row) for row in db.execute("SELECT * FROM events WHERE experiment=? ORDER BY id", (experiment_id,))
            ]

    def unregister_experiment(self, experiment_id: str) -> None:
        with self._transaction() as db:
            state = db.execute("SELECT * FROM states WHERE experiment=?", (experiment_id,)).fetchone()
            if state is None:
                return
            self._event(db, experiment_id, "deleted", state["current_version"])
            db.execute("UPDATE runtime SET experiment=NULL WHERE id=1 AND experiment=?", (experiment_id,))
            # Keep version/audit history so recreating an ID cannot reuse a
            # revision still held by an in-flight request from the old run.
            for table in ("states", "observations"):
                db.execute(f"DELETE FROM {table} WHERE experiment=?", (experiment_id,))
