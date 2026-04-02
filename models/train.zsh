#!/usr/bin/env zsh
# ── WeatherPredict Model Trainer ──────────────────────────────────────────────
# Usage:
#   ./train.zsh              # Interactive menu
#   ./train.zsh bottom_up    # Phase 1–5 full pipeline
#   ./train.zsh prepare      # Phases 1–2 + inference → save cache (no training)
#   ./train.zsh phase5       # Phase 5 from cache (fast, no Spark, ~3 min)
#   ./train.zsh status       # Show model status
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/Volumes/ExternalSSD/StudyMaterials/252_SPARK/WeatherPredict"
MODELS_DIR="${PROJECT_DIR}/models"
CONDA_ENV="pyspark"

# Model artifacts (for "last trained" display)
typeset -A MODEL_ARTIFACTS MODEL_LABELS
MODEL_ARTIFACTS=(
    bottom_up "${MODELS_DIR}/lgbm_storm_classifier.pkl"
    prepare "${MODELS_DIR}/phase5_monthly_cache.parquet"
    phase5 "${MODELS_DIR}/phase5_ensemble.pkl"
)
MODEL_LABELS=(
    bottom_up "Phase 1–5 — Full Pipeline"
    prepare "Prepare — Monthly Data Cache"
    phase5 "Phase 5 — Monthly Stacked Ensemble (from cache)"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
_bold()  { print -P "%B$1%b" }
_green() { print -P "%F{green}$1%f" }
_red()   { print -P "%F{red}$1%f" }
_cyan()  { print -P "%F{cyan}$1%f" }
_dim()   { print -P "%F{240}$1%f" }

last_trained() {
    local artifact=$1
    if [[ -f "$artifact" ]]; then
        stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$artifact"
    else
        echo "never"
    fi
}

ensure_conda() {
    # Activate conda env if not already active
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
        _dim "  Activating conda env: ${CONDA_ENV}..."
        if (( ${+commands[conda]} )); then
            eval "$(conda shell.zsh hook 2>/dev/null)"
        elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        else
            _red "✗ Cannot find conda. Please activate '${CONDA_ENV}' manually."
            exit 1
        fi
        conda activate "$CONDA_ENV" || { _red "✗ Failed to activate ${CONDA_ENV}"; exit 1; }
        _green "  ✓ Conda env '${CONDA_ENV}' active"
    else
        _dim "  Conda env '${CONDA_ENV}' already active"
    fi
}

train_model() {
    local key=$1
    local label="${MODEL_LABELS[$key]}"
    local artifact="${MODEL_ARTIFACTS[$key]}"

    echo ""
    _bold "━━━ ${label} ━━━"
    _dim "  Module:  models.bottom_up_forecast"
    _dim "  Last trained: $(last_trained "$artifact")"
    echo ""

    local start_time=$SECONDS
    python -u -m models.bottom_up_forecast "${@:2}"
    local elapsed=$(( SECONDS - start_time ))

    echo ""
    _green "  ✓ ${label} completed in ${elapsed}s"
    _dim "  Model saved: $(last_trained "$artifact")"
}

# ── Cluster helpers ───────────────────────────────────────────────────────────

detect_lan_ip() {
    # macOS: try en0 (WiFi), en1 (Ethernet)
    local ip
    ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")
    if [[ -z "$ip" ]]; then
        # Linux fallback
        ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
    fi
    echo "$ip"
}

resolve_spark_home() {
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
    fi
    if [[ -z "${SPARK_HOME:-}" || ! -d "$SPARK_HOME/sbin" ]]; then
        _red "✗ Cannot find SPARK_HOME. Set it manually: export SPARK_HOME=/path/to/spark"
        exit 1
    fi
    export SPARK_HOME
}

wait_for_workers() {
    local master_ip=$1
    local timeout=${2:-120}
    local elapsed=0
    _dim "  Waiting for workers to register (timeout: ${timeout}s)..."
    while (( elapsed < timeout )); do
        # Query Spark Master REST API for alive workers
        local workers
        workers=$(curl -s "http://${master_ip}:8080/json/" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    alive = [w for w in data.get('workers', []) if w.get('state') == 'ALIVE']
    print(len(alive))
except: print(0)" 2>/dev/null || echo 0)
        if (( workers >= 1 )); then
            _green "  ✓ ${workers} worker(s) registered!"
            return 0
        fi
        sleep 3
        elapsed=$(( elapsed + 3 ))
        printf "\r  ⏳ Waiting... %ds / %ds (workers: %s)" "$elapsed" "$timeout" "$workers"
    done
    echo ""
    _red "  ✗ No workers registered within ${timeout}s."
    _dim "  Run on the worker machine: ./cluster/start_worker.sh ${master_ip}"
    return 1
}

train_model_cluster() {
    local key=$1
    local label="${MODEL_LABELS[$key]}"
    local artifact="${MODEL_ARTIFACTS[$key]}"

    echo ""
    _bold "━━━ 🌐 CLUSTER: ${label} ━━━"

    # Detect LAN IP
    local master_ip
    master_ip=$(detect_lan_ip)
    if [[ -z "$master_ip" ]]; then
        _red "  ✗ Cannot detect LAN IP. Are you connected to a network?"
        exit 1
    fi
    _cyan "  Master IP: ${master_ip}"

    # Start Spark Master
    resolve_spark_home
    _dim "  Starting Spark Master..."
    export SPARK_MASTER_HOST="$master_ip"
    "$SPARK_HOME/sbin/start-master.sh" -h "$master_ip" -p 7077 --webui-port 8080 2>&1 | tail -1
    _green "  ✓ Master started at spark://${master_ip}:7077"
    _dim "  Web UI: http://${master_ip}:8080"

    # Also start a LOCAL worker on this machine (use ~50% resources for worker)
    local local_cores
    local_cores=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
    local worker_cores=$(( local_cores / 2 ))
    (( worker_cores < 2 )) && worker_cores=2
    local worker_mem="10g"  # leave ~14g for driver on 24GB machine

    _dim "  Starting local worker (${worker_cores} cores, ${worker_mem} RAM)..."
    "$SPARK_HOME/sbin/start-worker.sh" "spark://${master_ip}:7077" \
        -c "$worker_cores" -m "$worker_mem" 2>&1 | tail -1
    _green "  ✓ Local worker started"

    # Wait for at least 1 worker to register
    echo ""
    if ! wait_for_workers "$master_ip" 60; then
        _red "  ✗ Cluster setup failed. Stopping master."
        "$SPARK_HOME/sbin/stop-worker.sh" 2>/dev/null || true
        "$SPARK_HOME/sbin/stop-master.sh" 2>/dev/null || true
        exit 1
    fi

    # Run training with cluster config
    echo ""
    _dim "  Module:  models.bottom_up_forecast"
    _dim "  Last trained: $(last_trained "$artifact")"
    echo ""

    local start_time=$SECONDS
    SPARK_CLUSTER_MODE=cluster SPARK_MASTER_IP="$master_ip" python -u -m models.bottom_up_forecast
    local elapsed=$(( SECONDS - start_time ))

    echo ""
    _green "  ✓ ${label} completed in ${elapsed}s (cluster mode)"
    _dim "  Model saved: $(last_trained "$artifact")"

    # Stop cluster daemons
    echo ""
    _dim "  Stopping Spark cluster..."
    "$SPARK_HOME/sbin/stop-worker.sh" 2>/dev/null || true
    "$SPARK_HOME/sbin/stop-master.sh" 2>/dev/null || true
    _green "  ✓ Cluster stopped"
}

show_status() {
    echo ""
    _bold "╭─ Model Status ─────────────────────────────────────────╮"
    for key in bottom_up prepare phase5; do
        local artifact="${MODEL_ARTIFACTS[$key]}"
        local label="${MODEL_LABELS[$key]}"
        local trained="$(last_trained "$artifact")"
        if [[ "$trained" == "never" ]]; then
            printf "│  %-50s %s\n" "$label" "⚪ not trained"
        else
            printf "│  %-50s %s\n" "$label" "🟢 $trained"
        fi
    done
    _bold "╰────────────────────────────────────────────────────────╯"
    echo ""
}

show_menu() {
    show_status
    _bold "Select an option:"
    echo "  1) Phase 1–5: Complete Pipeline            (~45 min first run, ~20 min cached)"
    echo "  2) Prepare: Build monthly cache             (~20 min first run, ~5 min cached)"
    echo "  3) Phase 5: Train from cache                (~3 min, no Spark)"
    echo "  4) Show model status"
    echo ""
    echo "  ⚡ Cluster mode:"
    echo "  c) Phase 1–5 on Spark Standalone Cluster    (~25 min, 2 nodes)"
    echo ""
    _dim "  Note: Phase 1 features are cached permanently after first run."
    _dim "  To force recompute: ./train.zsh bottom_up --rebuild-features"
    echo ""
    echo "  q) Quit"
    echo ""
    printf "  Choice [1-4/c/q]: "
    read -r choice

    case "$choice" in
        1) SELECTION="bottom_up" ;;
        2) SELECTION="prepare" ;;
        3) SELECTION="phase5" ;;
        4) show_status; exit 0 ;;
        c|C) SELECTION="cluster" ;;
        q|Q) echo ""; exit 0 ;;
        *) _red "Invalid choice."; exit 1 ;;
    esac
}

# ── Main ──────────────────────────────────────────────────────────────────────
_bold "🌀 WeatherPredict Model Trainer"
_dim "  Project: ${PROJECT_DIR}"

# cd to project dir (required for `python -m models.bottom_up_forecast`)
if [[ "$PWD" != "$PROJECT_DIR" ]]; then
    cd "$PROJECT_DIR" || { _red "✗ Cannot cd to ${PROJECT_DIR}"; exit 1; }
    _dim "  Changed to: $PWD"
fi

# Activate conda
ensure_conda

# Determine what to train
SELECTION="${1:-}"
EXTRA_ARGS=("${@:2}")  # Pass extra flags through (e.g., --rebuild-features)
if [[ -z "$SELECTION" ]]; then
    show_menu
fi

# Run training
total_start=$SECONDS
case "$SELECTION" in
    bottom_up)
        train_model bottom_up "${EXTRA_ARGS[@]}"
        ;;
    prepare)
        train_model prepare --prepare "${EXTRA_ARGS[@]}"
        ;;
    phase5)
        train_model phase5 --phase5 "${EXTRA_ARGS[@]}"
        ;;
    cluster)
        train_model_cluster bottom_up
        ;;
    status)
        show_status
        exit 0
        ;;
    *)
        _red "Unknown option: ${SELECTION}"
        echo "Usage: $0 [bottom_up|prepare|phase5|cluster|status]"
        exit 1
        ;;
esac

total_elapsed=$(( SECONDS - total_start ))
echo ""
_green "━━━ All done in ${total_elapsed}s ━━━"
show_status
