#!/usr/bin/env bash
# ── WeatherPredict: Sync Data to Worker ───────────────────────────────────────
# Copies parquet_data/ and model artifacts to the worker machine via rsync.
# Requires passwordless SSH to the worker (ssh-copy-id first).
#
# Usage:
#   ./cluster/sync_data.sh <WORKER_USER> <WORKER_IP> <WORKER_PROJECT_DIR>
#   ./cluster/sync_data.sh fubuki 192.168.1.20 /Users/fubuki/WeatherPredict
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

WORKER_USER="${1:?Usage: $0 <WORKER_USER> <WORKER_IP> <WORKER_PROJECT_DIR>}"
WORKER_IP="${2:?Usage: $0 <WORKER_USER> <WORKER_IP> <WORKER_PROJECT_DIR>}"
WORKER_DIR="${3:?Usage: $0 <WORKER_USER> <WORKER_IP> <WORKER_PROJECT_DIR>}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKER_DEST="${WORKER_USER}@${WORKER_IP}:${WORKER_DIR}"

echo "📦 WeatherPredict Data Sync"
echo "   Source : $PROJECT_DIR"
echo "   Target : $WORKER_DEST"
echo ""

# ── Step 1: Ensure target directories exist ──
echo "1️⃣  Creating remote directories..."
ssh "${WORKER_USER}@${WORKER_IP}" "mkdir -p '${WORKER_DIR}/parquet_data' '${WORKER_DIR}/models'"

# ── Step 2: Sync parquet_data (the big one — ~15GB) ──
echo "2️⃣  Syncing parquet_data/ (this may take a while on first run)..."
rsync -avz --progress --compress-level=1 \
    "${PROJECT_DIR}/parquet_data/" \
    "${WORKER_DEST}/parquet_data/"

# ── Step 3: Sync model artifacts (small — a few MB) ──
echo "3️⃣  Syncing model artifacts..."
rsync -avz --progress \
    "${PROJECT_DIR}/models/"*.pkl \
    "${WORKER_DEST}/models/" 2>/dev/null || echo "   (no .pkl files to sync)"

# ── Step 4: Sync ONI data ──
echo "4️⃣  Syncing ONI data..."
rsync -avz --progress \
    "${PROJECT_DIR}/oni.csv" \
    "${WORKER_DEST}/" 2>/dev/null || true

# ── Step 5: Sync Python scripts (workers need these for mapInPandas) ──
echo "5️⃣  Syncing Python dependencies..."
rsync -avz --progress \
    "${PROJECT_DIR}/models/bottom_up_forecast.py" \
    "${WORKER_DEST}/models/"
rsync -avz --progress \
    "${PROJECT_DIR}/requirements.txt" \
    "${WORKER_DEST}/"

echo ""
echo "✅ Sync complete!"
echo ""
echo "Next steps on the worker machine:"
echo "  1. cd $WORKER_DIR"
echo "  2. conda activate pyspark"
echo "  3. pip install -r requirements.txt  (if first time)"
echo "  4. ./cluster/start_worker.sh <MASTER_IP>"
