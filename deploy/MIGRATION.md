# Kronos V2 — Cloud Server Migration Checklist

## Recommended timing
Do this **after** regime v2 deploys (~June 8-9) and you've confirmed it's running
cleanly for a day locally. Don't migrate infrastructure at the same time as a major
model change. Once stable, migrate (~June 10-11) and go live from the server.

## Server recommendation
- Provider: Hetzner CX22 (~€4/mo), DigitalOcean Basic ($6/mo), or Linode Nanode ($5/mo)
- Spec: 2 vCPU, 4GB RAM, 40GB SSD — Kronos model + Redis comfortably under 2GB
- OS: Ubuntu 22.04 LTS
- Region: pick low-latency to US East (Kalshi is US-based)

---

## Phase 1: Provision and configure server

- [ ] **Create VPS** — Ubuntu 22.04 LTS, 2 vCPU / 4GB RAM minimum
- [ ] **SSH in and update**
  ```bash
  apt update && apt upgrade -y
  ```
- [ ] **Create kronos user** (never run trading systems as root)
  ```bash
  useradd -m -s /bin/bash kronos
  # Optional: allow sudo for setup only
  usermod -aG sudo kronos
  ```
- [ ] **Install system dependencies**
  ```bash
  apt install -y python3.12 python3.12-venv python3.12-dev \
    redis-server git curl rsync build-essential
  ```
- [ ] **Enable and start Redis**
  ```bash
  systemctl enable redis-server
  systemctl start redis-server
  redis-cli ping   # should return PONG
  ```
- [ ] **Create app directory**
  ```bash
  mkdir -p /opt/kronos-v2
  chown kronos:kronos /opt/kronos-v2
  ```

---

## Phase 2: Transfer code and files

Run these from your Mac. Replace `SERVER_IP` with your server's IP address.

- [ ] **Clone or rsync the repo** (fastest: rsync, excludes secrets)
  ```bash
  rsync -avz --exclude='.env' --exclude='keys/' --exclude='venv/' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='logs/*.log' \
    "/Users/ezrakornberg/Kronos V2/" kronos@SERVER_IP:/opt/kronos-v2/
  ```
- [ ] **Transfer secrets securely** (NOT via git — these are private keys and API keys)
  ```bash
  scp "/Users/ezrakornberg/Kronos V2/.env" kronos@SERVER_IP:/opt/kronos-v2/.env
  scp -r "/Users/ezrakornberg/Kronos V2/keys/" kronos@SERVER_IP:/opt/kronos-v2/keys/
  ```
- [ ] **Transfer live database and models**
  ```bash
  # Stop local service first to avoid a mid-write copy
  launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.kronos.v2.plist
  
  scp "/Users/ezrakornberg/Kronos V2/trades.db" kronos@SERVER_IP:/opt/kronos-v2/trades.db
  rsync -avz "/Users/ezrakornberg/Kronos V2/models/" kronos@SERVER_IP:/opt/kronos-v2/models/
  
  # Restart local service (it's still your primary until server is confirmed working)
  launchctl load ~/Library/LaunchAgents/com.kronos.v2.plist
  ```
- [ ] **Fix permissions on server**
  ```bash
  ssh kronos@SERVER_IP "chmod 600 /opt/kronos-v2/.env /opt/kronos-v2/keys/*"
  ```

---

## Phase 3: Python environment

SSH in as the kronos user for these steps.

- [ ] **Create venv and install dependencies**
  ```bash
  cd /opt/kronos-v2
  python3.12 -m venv venv
  venv/bin/pip install --upgrade pip
  venv/bin/pip install -r requirements.txt
  ```
- [ ] **Verify Kronos model downloads** (first run pulls ~500MB from HuggingFace)
  ```bash
  venv/bin/python3 -c "
  from btc_kalshi_system.models.kronos_engine import KronosEngine
  e = KronosEngine(); e.preload(); print('Kronos model loaded OK')
  "
  ```
  This will download `NeoQuasar/Kronos-small` to the HuggingFace cache (~/.cache/huggingface).
  Takes 1-2 minutes. Subsequent starts are instant.

- [ ] **Create logs directory**
  ```bash
  mkdir -p /opt/kronos-v2/logs
  ```

---

## Phase 4: Install systemd service

- [ ] **Copy service file**
  ```bash
  cp /opt/kronos-v2/deploy/kronos-v2.service /etc/systemd/system/kronos-v2.service
  systemctl daemon-reload
  ```
- [ ] **Test start in paper mode** (verify `.env` has `PAPER_TRADING=true`)
  ```bash
  cat /opt/kronos-v2/.env | grep PAPER_TRADING   # must be true before first start
  systemctl start kronos-v2
  systemctl status kronos-v2
  ```
- [ ] **Tail logs to confirm startup**
  ```bash
  journalctl -u kronos-v2 -f
  ```
  Expected within 60s:
  - `KronosEngine: model ready on CPU`
  - `KalshiOrderbookFeed: WS connected`
  - `DerivativesFeed: wrote regime:features`
  - `[PAPER] Simulated fill:` on first trade

- [ ] **Enable on boot**
  ```bash
  systemctl enable kronos-v2
  ```

---

## Phase 5: Install cron jobs

- [ ] **Install cron jobs as kronos user**
  ```bash
  su - kronos
  crontab /opt/kronos-v2/deploy/crontab-linux.txt
  crontab -l   # verify
  ```

---

## Phase 6: Smoke test (run for 1-2 hours in paper mode)

- [ ] **Confirm WS orderbook running** — no REST fallback after first 30s
  ```bash
  journalctl -u kronos-v2 --since "5 minutes ago" | grep -E "WS orderbook|REST fallback"
  ```
- [ ] **Confirm features writing to Redis**
  ```bash
  redis-cli get regime:features | python3 -m json.tool | head -20
  ```
- [ ] **Confirm paper trades firing and logging to DB**
  ```bash
  cd /opt/kronos-v2
  venv/bin/python3 -c "
  import sqlite3
  db = sqlite3.connect('trades.db')
  print(db.execute('SELECT COUNT(*) FROM trades').fetchone())
  print(db.execute('SELECT timestamp, direction, fill_price_cents FROM trades ORDER BY timestamp DESC LIMIT 3').fetchall())
  "
  ```
- [ ] **Confirm candle_features logging**
  ```bash
  cd /opt/kronos-v2
  venv/bin/python3 -c "
  import sqlite3
  db = sqlite3.connect('trades.db')
  r = db.execute('SELECT COUNT(*), MAX(logged_at) FROM candle_features').fetchone()
  print(f'{r[0]} rows, last logged: {r[1]}')
  "
  ```

---

## Phase 7: Decommission local Mac service

Only do this once the server has been running cleanly for 24+ hours.

- [ ] **Stop and unload local service**
  ```bash
  launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.kronos.v2.plist
  ```
- [ ] **Final DB sync** (get any trades that fired while both were running — shouldn't be any since you stopped local first, but verify)
  ```bash
  # On server: check row count
  ssh kronos@SERVER_IP "cd /opt/kronos-v2 && venv/bin/python3 -c \"import sqlite3; print(sqlite3.connect('trades.db').execute('SELECT COUNT(*) FROM trades').fetchone())\""
  ```
- [ ] **Remove local cron jobs**
  ```bash
  crontab -l   # verify they exist
  crontab -r   # remove all — re-add any non-Kronos crons manually if needed
  ```

---

## Going live from the server

When ready (edge confirmed + P&L positive, ~June 15-18):

```bash
ssh kronos@SERVER_IP
cd /opt/kronos-v2

# Edit .env
sed -i 's/PAPER_TRADING=true/PAPER_TRADING=false/' .env

# Restart service
systemctl restart kronos-v2

# Confirm live order in logs
journalctl -u kronos-v2 -f | grep "Order placed"
```

---

## Useful server commands

```bash
# View live logs
journalctl -u kronos-v2 -f

# Restart service (e.g. after code update)
systemctl restart kronos-v2

# Check service status
systemctl status kronos-v2

# Pause regime model (drawdown protection)
touch /opt/kronos-v2/models/regime_paused.flag
systemctl restart kronos-v2

# Deploy code update from Mac
rsync -avz --exclude='.env' --exclude='keys/' --exclude='venv/' \
  --exclude='__pycache__' --exclude='*.pyc' \
  "/Users/ezrakornberg/Kronos V2/" kronos@SERVER_IP:/opt/kronos-v2/
ssh kronos@SERVER_IP "systemctl restart kronos-v2"

# Run regime dry-run from server
ssh kronos@SERVER_IP "cd /opt/kronos-v2 && source .env && venv/bin/python3 scripts/train_regime.py --dry-run"
```
