#!/usr/bin/env bash
# ── WeatherPredict: Stop Spark Cluster ────────────────────────────────────────
# Stops both Master and Worker daemons.
# Safe to run on either machine — only stops what's running locally.
#
# Usage:
#   ./cluster/stop_cluster.sh          # Stop all local daemons
#   ./cluster/stop_cluster.sh master   # Stop master only
#   ./cluster/stop_cluster.sh worker   # Stop worker only
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

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

TARGET="${1:-all}"

case "$TARGET" in
    master)
        echo "🛑 Stopping Spark Master..."
        "$SPARK_HOME/sbin/stop-master.sh" 2>&1 || true
        echo "✅ Master stopped."
        ;;
    worker)
        echo "🛑 Stopping Spark Worker..."
        "$SPARK_HOME/sbin/stop-worker.sh" 2>&1 || true
        echo "✅ Worker stopped."
        ;;
    all)
        echo "🛑 Stopping all Spark daemons..."
        "$SPARK_HOME/sbin/stop-worker.sh" 2>&1 || true
        "$SPARK_HOME/sbin/stop-master.sh" 2>&1 || true
        echo "✅ All Spark daemons stopped."
        ;;
    *)
        echo "Usage: $0 [master|worker|all]"
        exit 1
        ;;
esac

# Clean up any stale PID files
SPARK_PID_DIR="${SPARK_PID_DIR:-/tmp}"
for pid_file in "$SPARK_PID_DIR"/spark-*.pid; do
    if [[ -f "$pid_file" ]]; then
        PID=$(cat "$pid_file")
        if ! kill -0 "$PID" 2>/dev/null; then
            rm -f "$pid_file"
            echo "   Cleaned stale PID file: $(basename "$pid_file")"
        fi
    fi
done
