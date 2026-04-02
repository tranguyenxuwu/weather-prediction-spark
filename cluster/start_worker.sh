#!/usr/bin/env bash
# ── WeatherPredict: Start Spark Worker ────────────────────────────────────────
# Run this on the WORKER (secondary) machine.
# Connects to the Spark Master running on the primary machine.
#
# Usage:
#   ./cluster/start_worker.sh <MASTER_IP>
#   ./cluster/start_worker.sh 192.168.1.10
#   ./cluster/start_worker.sh 192.168.1.10 8 16g   # custom cores/memory
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Args ──
MASTER_IP="${1:?Usage: $0 <MASTER_IP> [CORES] [MEMORY]}"
WORKER_CORES="${2:-}"
WORKER_MEMORY="${3:-}"

MASTER_PORT=7077
MASTER_URL="spark://$MASTER_IP:$MASTER_PORT"

# ── Resolve SPARK_HOME ──
if [[ -z "${SPARK_HOME:-}" ]]; then
    if command -v brew &>/dev/null; then
        SPARK_HOME="$(brew --prefix apache-spark 2>/dev/null)/libexec" || true
    fi
    for candidate in /opt/spark /usr/local/spark "$HOME/spark"; do
        if [[ -d "$candidate/sbin" ]]; then
            SPARK_HOME="$candidate"
            break
        fi
    done
    if [[ -z "${SPARK_HOME:-}" || ! -d "$SPARK_HOME/sbin" ]]; then
        echo "✗ Cannot find SPARK_HOME. Set it manually: export SPARK_HOME=/path/to/spark"
        exit 1
    fi
fi
export SPARK_HOME

# ── Auto-detect resources if not specified ──
if [[ -z "$WORKER_CORES" ]]; then
    WORKER_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

if [[ -z "$WORKER_MEMORY" ]]; then
    # Use ~75% of total system RAM for the worker
    if command -v free &>/dev/null; then
        TOTAL_MB=$(free -m | awk '/Mem:/{print $2}')
    elif command -v sysctl &>/dev/null; then
        TOTAL_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 8589934592)
        TOTAL_MB=$((TOTAL_BYTES / 1048576))
    else
        TOTAL_MB=8192
    fi
    WORKER_MB=$((TOTAL_MB * 3 / 4))
    WORKER_MEMORY="${WORKER_MB}m"
fi

# ── Pre-flight check: can we reach the master? ──
echo "🔍 Checking connectivity to master at $MASTER_IP..."
if ping -c 1 -W 2 "$MASTER_IP" &>/dev/null; then
    echo "   ✓ Master reachable"
else
    echo "   ⚠️  Cannot ping $MASTER_IP — check network. Attempting start anyway..."
fi

# ── Start Worker ──
echo ""
echo "🚀 Starting Spark Worker..."
echo "   SPARK_HOME  : $SPARK_HOME"
echo "   Master URL  : $MASTER_URL"
echo "   Worker Cores: $WORKER_CORES"
echo "   Worker Mem  : $WORKER_MEMORY"
echo ""

"$SPARK_HOME/sbin/start-worker.sh" "$MASTER_URL" \
    -c "$WORKER_CORES" \
    -m "$WORKER_MEMORY" \
    2>&1

echo ""
echo "✅ Spark Worker started!"
echo "   Connected to: $MASTER_URL"
echo "   Worker UI   : http://$(hostname -I 2>/dev/null | awk '{print $1}' || ipconfig getifaddr en0 2>/dev/null || echo localhost):8081"
echo ""
echo "Verify registration at: http://$MASTER_IP:8080"
