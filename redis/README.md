# Redis High Availability (Sentinel & Master-Replica Setup)

This directory contains the production-ready configuration, topology definitions, and client integration guidelines for **Redis High Availability (HA)** using **Redis Sentinel** and **Master-Replica Replication**.

---

## 1. Architecture Overview

Redis Sentinel provides high availability for Redis without requiring complex clustering sharding. It handles:

1. **Monitoring**: Continuously checks if master and replica instances are functioning as expected.
2. **Notification**: Emits alerts and events (via Pub/Sub and scripts) when an issue occurs.
3. **Automatic Failover**: When a primary master node fails, Sentinels vote and automatically promote one of the read replicas to become the new master.
4. **Configuration Provider**: Acts as an authoritative source of service discovery for clients, returning the address of the current active master and available read replicas.

### HA Topology Diagram

```
                 +-----------------------+
                 |  Deen Bridge AI App   |
                 +-----------+-----------+
                             |
                   Queries Current Master
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
+-----------------+ +-----------------+ +-----------------+
|   Sentinel 1    | |   Sentinel 2    | |   Sentinel 3    |
|   (Port 26379)  | |   (Port 26380)  | |   (Port 26381)  |
+--------+--------+ +--------+--------+ +--------+--------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                     Monitors / Promotes
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
+-----------------+   Replication (Async)   +-----------------+
|  Redis Master   | ----------------------> | Redis Replica 1 |
|  (Port 6379)    |                         |  (Port 6380)    |
+--------+--------+                         +-----------------+
         |                                       ^
         |            Replication (Async)        |
         +---------------------------------------+
         |
         v
+-----------------+
| Redis Replica 2 |
|  (Port 6381)    |
+-----------------+
```

### Quorum & Fault Tolerance

* **Topology**: 1 Master + 2 Replicas + 3 Sentinels.
* **Quorum = 2**: At least 2 Sentinels must agree that the master is unreachable (`ODOWN`) to trigger an automated failover.
* **Resilience**: The cluster can withstand the failure of any 1 Redis node AND any 1 Sentinel simultaneously without downtime.

---

## 2. Configuration Files

| File | Purpose |
|------|---------|
| [`redis.conf`](redis.conf) | Redis server configuration for master and replica nodes (AOF+RDB persistence, memory limits, replication). |
| [`sentinel.conf`](sentinel.conf) | Sentinel service configuration (quorum=2, 5s timeout, auto-failover rules). |
| [`docker-compose.yml`](docker-compose.yml) | Multi-container Docker Compose deployment simulating the full 6-node HA topology. |

### Key Configuration Parameters

#### In `redis.conf`:
* `appendonly yes` & `appendfsync everysec`: Hybrid AOF + RDB persistence for minimal data loss ($< 1\text{s}$).
* `maxmemory 512mb` & `maxmemory-policy allkeys-lru`: Least Recently Used key eviction under memory pressure.
* `replica-serve-stale-data yes`: Replicas continue answering read requests during disconnections or resynchronization.
* `replica-read-only yes`: Enforces data integrity on replica nodes.

#### In `sentinel.conf`:
* `sentinel monitor mymaster redis-master 6379 2`: Defines the master cluster name and quorum.
* `sentinel down-after-milliseconds mymaster 5000`: 5 seconds of unresponsiveness marks a node as Subjectively Down (`SDOWN`).
* `sentinel failover-timeout mymaster 60000`: 60-second limit before an uncompleted failover is retried.
* `sentinel parallel-syncs mymaster 1`: Replicas reconfigure to the new master one by one to preserve read availability.

---

## 3. Quickstart with Docker Compose

Start the full HA cluster (1 master, 2 replicas, 3 sentinels):

```bash
cd redis
docker compose up -d
```

Verify all containers are healthy:

```bash
docker compose ps
```

---

## 4. Client Connection Configuration (Python)

To connect the application to the Sentinel cluster, use `redis.asyncio.sentinel.Sentinel` from the `redis` Python package.

### Connection Pattern

```python
import os
from redis.asyncio import Redis
from redis.asyncio.sentinel import Sentinel

# 1. Define Sentinel hosts and Master name
SENTINEL_HOSTS = [
    ("localhost", 26379),
    ("localhost", 26380),
    ("localhost", 26381),
]
MASTER_NAME = os.getenv("REDIS_SENTINEL_MASTER_NAME", "mymaster")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# 2. Instantiate Sentinel client
sentinel = Sentinel(
    SENTINEL_HOSTS,
    sentinel_kwargs={"password": REDIS_PASSWORD},
    password=REDIS_PASSWORD,
    decode_responses=True,
)

# 3. Obtain Master connection for writes and durable stores
async def get_master_client() -> Redis:
    """Returns a client connected to the current active master."""
    return sentinel.master_for(
        MASTER_NAME,
        socket_timeout=2.0,
        retry_on_timeout=True,
    )

# 4. Obtain Replica connection for read-heavy operations (optional)
async def get_replica_client() -> Redis:
    """Returns a client connected to an active read replica."""
    return sentinel.slave_for(
        MASTER_NAME,
        socket_timeout=2.0,
        retry_on_timeout=True,
    )
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `REDIS_USE_SENTINEL` | Set to `true` to enable Sentinel service discovery | `true` |
| `REDIS_SENTINEL_HOSTS` | Comma-separated `host:port` pairs for Sentinels | `sentinel-1:26379,sentinel-2:26379,sentinel-3:26379` |
| `REDIS_SENTINEL_MASTER_NAME` | Master group name monitored by Sentinels | `mymaster` |
| `REDIS_URL` | Fallback standalone URL when Sentinel is disabled | `redis://localhost:6379/0` |
| `REDIS_PASSWORD` | Optional authentication password | `your-secret-password` |

---

## 5. Failover Behavior & Testing

### How Failover Works Step-by-Step

1. **Failure Detection**: Master stops responding to PINGs for $\ge 5000\text{ms}$.
2. **Subjective Down (`SDOWN`)**: Each Sentinel independently marks the master down.
3. **Objective Down (`ODOWN`)**: Once 2 Sentinels (Quorum) agree, the master enters `ODOWN`.
4. **Leader Election**: Sentinels vote using the Raft-like election protocol to elect one Sentinel as the failover coordinator.
5. **Replica Selection**: The leader Sentinel chooses the best replica based on:
   - Lowest `replica-priority`
   - Most up-to-date replication offset
   - Lowest run ID (tie-breaker)
6. **Promotion & Reconfiguration**:
   - The selected replica receives `SLAVEOF NO ONE` and becomes the new master.
   - The remaining replicas are reconfigured to replicate from the new master.
   - If the old master comes back online later, Sentinels automatically reconfigure it as a replica of the new master.

### Testing Automated Failover

1. Check current master info via Sentinel:
   ```bash
   docker exec -it redis-sentinel-1 redis-cli -p 26379 sentinel get-master-addr-by-name mymaster
   # Output: 172.x.x.x 6379 (points to redis-master)
   ```

2. Pause or stop the current master:
   ```bash
   docker stop redis-master
   ```

3. Watch Sentinel logs:
   ```bash
   docker logs -f redis-sentinel-1
   # You will see:
   # +sdown master mymaster ...
   # +odown master mymaster ... #quorum 2/2
   # +try-failover master mymaster ...
   # +switch-master mymaster ...
   ```

4. Verify the new master address:
   ```bash
   docker exec -it redis-sentinel-2 redis-cli -p 26379 sentinel get-master-addr-by-name mymaster
   # Output: Points to redis-replica-1 (port 6380) or redis-replica-2 (port 6381)
   ```

5. Recover the old master:
   ```bash
   docker start redis-master
   # Sentinel detects it and automatically demotes it to a replica of the new master.
   ```

---

## 6. Monitoring & Diagnostic Commands

Run these commands using `redis-cli`:

### Inspect Replication Status on a Redis Node:
```bash
redis-cli -p 6379 info replication
```

### Inspect Sentinel Master Health:
```bash
redis-cli -p 26379 sentinel master mymaster
```

### List Replicas Monitored by Sentinel:
```bash
redis-cli -p 26379 sentinel slaves mymaster
```

### List Active Sentinel Peers:
```bash
redis-cli -p 26379 sentinel sentinels mymaster
```

### Trigger Manual Failover (for maintenance):
```bash
redis-cli -p 26379 sentinel failover mymaster
```

---

## 7. Security Best Practices for Production

1. **Authentication**:
   - Set `requirepass <password>` and `masterauth <password>` in `redis.conf`.
   - Set `sentinel auth-pass mymaster <password>` in `sentinel.conf`.
2. **Network Isolation**:
   - Do not expose Redis ports (`6379`, `26379`) to public internet interfaces.
   - Bind to internal VPC or private subnet interfaces (`bind 10.x.x.x` or `bind 0.0.0.0` within isolated Docker networks).
3. **Protected Mode**:
   - Keep `protected-mode yes` when binding without passwords.
4. **Script Protection**:
   - Keep `sentinel deny-scripts-reconfig yes` enabled to prevent unauthorized runtime script modifications.
