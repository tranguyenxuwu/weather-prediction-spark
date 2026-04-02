# Cluster Setup Guide: 2-Node Spark Standalone

This guide walks through setting up a 2-node Spark Standalone cluster for
faster WeatherPredict pipeline training.

**Your setup:**
- **Master:** Mac (Apple M4, 24GB RAM) — runs Spark Driver + Master + local Executor
- **Worker:** Windows 11 (Ryzen 5 3600, 16GB RAM) — runs Spark Worker + Executor

---

## Architecture

```
┌─────────────────────────────────────┐    LAN     ┌──────────────────────────────┐
│  MASTER NODE (Mac M4, 24GB)         │◄──────────►│  WORKER NODE (Win11, 16GB)   │
│                                     │  WiFi/     │                              │
│  Spark Master  (:7077)              │  Ethernet  │  Spark Worker                │
│  Spark Driver  (10GB RAM)           │            │  Executor (10GB, 5 cores)    │
│  Executor 1    (10GB, 5 cores)      │            │  parquet_data/ (copied)      │
│  parquet_data/ (ExternalSSD)        │            │                              │
└─────────────────────────────────────┘            └──────────────────────────────┘
```

**Why 2 nodes?** The pipeline processes ~600M rows. On a single 24GB machine,
Spark spills to disk continuously (Phase 2: 76 min, Phase 5: 124 min). Adding a
second machine doubles available RAM and CPU, eliminating most spills.

---

## Prerequisites

### Both Machines

| Requirement | Mac (Master) | Windows (Worker) |
|-------------|:---:|:---:|
| **Spark 4.1.1** | `brew install apache-spark` | Via Spark download + Hadoop winutils |
| **Java 21+** | `brew install openjdk@21` | Download from adoptium.net |
| **Python 3.11+** | Conda env `pyspark` | Conda env `pyspark` |
| **Same packages** | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| **Network** | Same LAN, ports 7077/8080 open | Same LAN, firewall allows Spark |

---

## Part 1: Setting Up the Windows Worker

### Step 1.1: Install Java

1. Download **Adoptium JDK 21** from https://adoptium.net/
2. Install it, checking "Set JAVA_HOME" during installation
3. Verify:
   ```powershell
   java -version
   # Should show: openjdk version "21.x.x"
   ```

### Step 1.2: Install Spark

1. Download Spark 4.1.1 (Pre-built for Hadoop 3):
   https://spark.apache.org/downloads.html

2. Extract to `C:\spark` (or any path **without spaces**)

3. Download **winutils.exe** for Hadoop 3:
   https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin

4. Place `winutils.exe` in `C:\spark\bin\`

5. Set environment variables (System → Advanced → Environment Variables):
   ```
   SPARK_HOME = C:\spark
   HADOOP_HOME = C:\spark
   ```
   Add to PATH: `C:\spark\bin`

6. Verify:
   ```powershell
   spark-submit --version
   # Should show: version 4.1.1
   ```

### Step 1.3: Install Python Environment

```powershell
# Install Miniconda if you don't have it
# Download from: https://docs.conda.io/en/latest/miniconda.html

conda create -n pyspark python=3.11 -y
conda activate pyspark
pip install -r requirements.txt
```

> ⚠️ **Important:** The Python version must match exactly between Mac and Windows.
> Check with `python --version` on both machines.

### Step 1.4: Open Firewall Ports

```powershell
# Run PowerShell as Administrator
netsh advfirewall firewall add rule name="Spark Worker" dir=in action=allow protocol=TCP localport=7077-7100
netsh advfirewall firewall add rule name="Spark Worker UI" dir=in action=allow protocol=TCP localport=8080-8082
netsh advfirewall firewall add rule name="Spark Executor" dir=in action=allow protocol=TCP localport=0-65535

# Or simply: Settings → Windows Firewall → Turn off for Private Network (less secure)
```

---

## Part 2: Transferring Data to Windows Worker

The `parquet_data/` directory (~15GB) must be accessible on both machines.
Here are three methods, from easiest to fastest:

### Method A: USB / External Drive (Simplest)

1. Copy `parquet_data/` folder to a USB drive
2. Plug into Windows machine
3. Copy to `C:\WeatherPredict\parquet_data\`

```
Total transfer time: ~10 minutes (USB 3.0)
```

### Method B: Network File Copy via `scp` (Over WiFi/Ethernet)

**On Mac (sender):**

```bash
# Find your Windows machine's IP
# On Windows: ipconfig → look for IPv4 Address (e.g., 192.168.1.20)

# Install OpenSSH on Windows first (see below), then:
scp -r /Volumes/ExternalSSD/StudyMaterials/252_SPARK/WeatherPredict/parquet_data/ \
    user@192.168.1.20:C:/WeatherPredict/parquet_data/
```

**Enable SSH on Windows (one-time):**
1. Settings → Apps → Optional Features → Add a Feature
2. Search "OpenSSH Server" → Install
3. Open PowerShell as Admin:
   ```powershell
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   ```

```
Transfer time: ~15 min (Ethernet) / ~45 min (WiFi)
```

### Method C: Shared Folder (Drag & Drop)

1. **On Windows:** Right-click `C:\WeatherPredict` → Properties → Sharing → Share
2. **On Mac:** Finder → Go → Connect to Server → `smb://192.168.1.20/WeatherPredict`
3. Drag `parquet_data/` folder into the shared location

```
Transfer time: ~20 min (Ethernet)
```

### Method D: `rsync` via WSL (Fastest Repeated Syncs)

If you have WSL (Windows Subsystem for Linux) installed:

**On Mac:**
```bash
# Use the sync script (rsync via WSL)
./cluster/sync_data.sh <WINDOWS_USER> <WINDOWS_IP> /mnt/c/WeatherPredict
```

**Install WSL on Windows (one-time):**
```powershell
# PowerShell as Admin
wsl --install -d Ubuntu
# After restart, set up username/password
# Then install rsync:
sudo apt update && sudo apt install rsync openssh-server -y
sudo service ssh start
```

> **Recommendation:** For first-time setup, use **Method A** (USB) or **Method C** (shared folder).
> For repeated syncs after model updates, use **Method D** (rsync via WSL).

---

## Part 3: Starting the Cluster

### Automated (Recommended — runs from Mac)

```bash
# This starts master + local worker, waits for remote worker, trains, stops
./models/train.zsh cluster
```

Meanwhile on the Windows machine, start the worker:

```powershell
cd C:\spark
.\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker spark://<MAC_IP>:7077 -c 10 -m 10g
```

### Manual Step-by-Step

**Step 1: Start Master (on Mac)**
```bash
./cluster/start_master.sh
# Note the Master URL: spark://<MAC_IP>:7077
```

**Step 2: Start Worker (on Windows)**
```powershell
# PowerShell — start Spark Worker
conda activate pyspark
cd C:\spark

# Connect to the Mac's Spark Master
.\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker spark://<MAC_IP>:7077 -c 10 -m 10g

# -c 10 = use 10 of the Ryzen's 12 threads
# -m 10g = allocate 10GB RAM to Spark (leaves 6GB for Windows)
```

**Step 3: Verify**
- Open browser: `http://<MAC_IP>:8080`
- Should show 2 Workers (1 Mac, 1 Windows) as ALIVE

**Step 4: Run Training (on Mac)**
```bash
SPARK_CLUSTER_MODE=cluster SPARK_MASTER_IP=<MAC_IP> python -u models/bottom_up_forecast.py
```

**Step 5: Stop Cluster**
```bash
# On Mac:
./cluster/stop_cluster.sh

# On Windows: Ctrl+C in the PowerShell window running the worker
```

---

## Part 4: Configuration Tuning

### Memory Allocation

Edit `models/bottom_up_forecast.py` → `SPARK_CONFIG_CLUSTER`:

| Setting | Default | Your Setup |
|---------|---------|------------|
| `spark.driver.memory` | `10g` | 10g (Mac has 24GB, leaves 14GB for executors) |
| `spark.executor.memory` | `10g` | 10g (Windows: 16GB total, leaves 6GB for OS) |
| `spark.executor.cores` | `5` | 5 (Ryzen 5 3600 has 12 threads, 10 for worker) |
| `spark.default.parallelism` | `15` | 15 (Mac 5 cores + Win 10 threads) |

### Local Worker Resources (Mac side)

In `train.zsh`, the `train_model_cluster()` function starts a local worker:

```bash
local worker_cores=$(( local_cores / 2 ))  # 5 of 10 cores
local worker_mem="10g"                      # 10GB
```

### Windows Worker Resources

When starting the worker on Windows, adjust `-c` and `-m`:

```powershell
# Conservative (leaves more for Windows):
.\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker spark://<IP>:7077 -c 8 -m 8g

# Aggressive (maximum Spark performance):
.\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker spark://<IP>:7077 -c 11 -m 12g
```

---

## Part 5: Troubleshooting

### Worker can't connect to master

```bash
# On Mac: check master is running
curl http://<MAC_IP>:8080/json/

# On Windows: can you reach the Mac?
ping <MAC_IP>

# Firewall issues? Temporarily disable:
# Mac: System Settings → Network → Firewall → Turn Off
# Win: Settings → Windows Security → Firewall → Turn off for Private
```

### `winutils.exe` error on Windows

```
ERROR Shell: Failed to locate the winutils binary in the hadoop binary path
```

Fix: Download `winutils.exe` and set `HADOOP_HOME`:
```powershell
$env:HADOOP_HOME = "C:\spark"
# Also add permanently via System Environment Variables
```

### Python version mismatch

```
Py4JJavaError: Python worker exited unexpectedly
```

Both machines MUST use the same Python version:
```bash
# Mac:
python --version  # e.g., Python 3.11.8

# Windows:
python --version  # Must match: Python 3.11.x
```

### Module not found on Windows worker

```powershell
conda activate pyspark
pip install -r requirements.txt
```

### Data path mismatch

Spark executors read data locally. Ensure the **same relative path** structure:
- Mac: `/Volumes/ExternalSSD/.../WeatherPredict/parquet_data/`
- Win: `C:\WeatherPredict\parquet_data\`

> ⚠️ If paths differ, you may need to set `INPUT_PATH` via environment variable.
> Currently the pipeline uses `Path(__file__).parent.parent / "parquet_data"` which
> will resolve correctly on the master (Mac) driver.

### Slow performance / not faster than local

- Check Spark UI (`http://<MAC_IP>:4040`) → Stages tab → are tasks distributed?
- If all tasks run on one executor, the data might not be partitioned well
- Increase shuffle partitions: set `spark.sql.shuffle.partitions` to `200`

---

## Switching Back to Local Mode

The pipeline defaults to local mode. No env vars needed:

```bash
# Local (default):
python models/bottom_up_forecast.py
./models/train.zsh bottom_up

# Cluster (only when explicitly set):
SPARK_CLUSTER_MODE=cluster python models/bottom_up_forecast.py
./models/train.zsh cluster
```
