"""
Count total rows across all Parquet files in parquet_data/
Uses multithreading + live progress display.
"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq

BASE_DIR = "parquet_data"

# ── Shared state ──
lock = threading.Lock()
total_rows = 0
files_done = 0
total_files = 0
category_rows = {"ibtracs": 0, "era5": 0, "noaa_sst": 0}


def count_file(filepath: str, category: str) -> int:
    """Read parquet metadata to get row count (no data loading)."""
    global total_rows, files_done
    try:
        meta = pq.read_metadata(filepath)
        n = meta.num_rows
    except Exception:
        # Fallback: read the file
        try:
            table = pq.read_table(filepath, columns=[])
            n = table.num_rows
        except Exception as e:
            print(f"\n⚠️  Error reading {filepath}: {e}", file=sys.stderr)
            n = 0

    with lock:
        total_rows += n
        files_done += 1
        category_rows[category] += n
    return n


def collect_parquet_files(root_dir: str) -> list[tuple[str, str]]:
    """Walk directory tree and collect all .parquet files with their category."""
    result = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Determine category from path
        rel = os.path.relpath(dirpath, root_dir)
        if rel.startswith("ibtracs"):
            cat = "ibtracs"
        elif rel.startswith("era5"):
            cat = "era5"
        elif rel.startswith("noaa_sst"):
            cat = "noaa_sst"
        else:
            cat = "other"

        for f in filenames:
            if f.endswith(".parquet"):
                result.append((os.path.join(dirpath, f), cat))
    return result


def progress_printer():
    """Background thread that prints live progress every 0.5s."""
    while not stop_event.is_set():
        with lock:
            done = files_done
            total = total_files
            rows = total_rows
        pct = (done / total * 100) if total > 0 else 0
        bar_len = 40
        filled = int(bar_len * done / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  {bar} {pct:5.1f}% | {done:,}/{total:,} files | {rows:,} rows counted",
            end="", flush=True,
        )
        time.sleep(0.5)


if __name__ == "__main__":
    t0 = time.time()

    print(f"📂 Scanning {BASE_DIR}/ for .parquet files...")
    all_files = collect_parquet_files(BASE_DIR)
    total_files = len(all_files)
    print(f"   Found {total_files:,} parquet files\n")

    # Start progress printer
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=progress_printer, daemon=True)
    progress_thread.start()

    # Count rows using thread pool (metadata-only reads are I/O bound)
    MAX_WORKERS = os.cpu_count() * 4  # I/O bound, can use more threads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(count_file, fp, cat): (fp, cat)
            for fp, cat in all_files
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions

    # Stop progress printer
    stop_event.set()
    progress_thread.join()

    elapsed = time.time() - t0

    # Final summary
    print(f"\r{' ' * 100}\r", end="")  # clear line
    print("=" * 60)
    print("📊 PARQUET ROW COUNT SUMMARY")
    print("=" * 60)
    for cat, count in category_rows.items():
        if count > 0:
            print(f"  {cat:.<30s} {count:>15,} rows")
    print("-" * 60)
    print(f"  {'TOTAL':.<30s} {total_rows:>15,} rows")
    print(f"  {'Files processed':.<30s} {files_done:>15,}")
    print(f"  {'Time elapsed':.<30s} {elapsed:>14.1f}s")
    print("=" * 60)
