#!/usr/bin/env bash
# ── WeatherPredict: Start Spark Master ────────────────────────────────────────
# Run this on the PRIMARY machine (driver node).
# The master coordinates work distribution across the cluster.
#
# Usage:
#   ./cluster/start_master.sh           # Auto-detect LAN IP
#   ./cluster/start_master.sh 192.168.1.10  # Explicit IP
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve SPARK_HOME ──
if [[ -z "${SPARK_HOME:-}" ]]; then
    # macOS Homebrew default
    if command -v brew &>/dev/null; then
        SPARK_HOME="$(brew --prefix apache-spark 2>/dev/null)/libexec" || true
    fi
    # Fallback: check common paths
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

# ── Determine Master IP ──
if [[ -n "${1:-}" ]]; then
    MASTER_IP="$1"
else
    # Auto-detect LAN IP (en0 = WiFi, en1 = Ethernet on macOS)
    MASTER_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
fi

if [[ "$MASTER_IP" == "127.0.0.1" ]]; then
    echo "⚠️  Could not detect LAN IP. Using localhost — workers on other machines cannot connect."
fi

export SPARK_MASTER_HOST="$MASTER_IP"
MASTER_PORT=7077
MASTER_WEBUI_PORT=8080

# ── Start Master ──
echo "🚀 Starting Spark Master..."
echo "   SPARK_HOME : $SPARK_HOME"
echo "   Master Host: $MASTER_IP"
echo "   Master Port: $MASTER_PORT"
echo "   Web UI Port: $MASTER_WEBUI_PORT"
echo ""

"$SPARK_HOME/sbin/start-master.sh" \
    -h "$MASTER_IP" \
    -p "$MASTER_PORT" \
    --webui-port "$MASTER_WEBUI_PORT" \
    2>&1

echo ""
echo "✅ Spark Master started!"
echo "   Master URL : spark://$MASTER_IP:$MASTER_PORT"
echo "   Web UI     : http://$MASTER_IP:$MASTER_WEBUI_PORT"
echo ""
echo "Next steps:"
echo "  1. On the WORKER machine, run:"
echo "     ./cluster/start_worker.sh $MASTER_IP"
echo "  2. Verify at http://$MASTER_IP:$MASTER_WEBUI_PORT"
echo "  3. Start training with: ./models/train.zsh cluster"
