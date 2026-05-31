#!/usr/bin/env bash
# Kronos live monitor — run in a separate terminal tab
# Refreshes every 30 seconds. Ctrl+C to stop.

DB="/Users/ezrakornberg/Kronos V2/trades.db"
LOG_DIR="/Users/ezrakornberg/Kronos V2/logs"
CAL_META="/Users/ezrakornberg/Kronos V2/models/calibrator_last_trained.json"

# ── Colors ────────────────────────────────────────────────────────────────────
RESET='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
CYAN='\033[36m'
WHITE='\033[37m'
MAGENTA='\033[35m'

BG_DARK='\033[40m'

color_prob() {
  local val="$1"
  local num
  num=$(echo "$val" | grep -oE '[0-9]+\.?[0-9]*' | head -1)
  if [ -z "$num" ]; then echo -e "${DIM}N/A${RESET}"; return; fi
  if (( $(echo "$num >= 0.70" | bc -l) )); then
    echo -e "${BOLD}${GREEN}$val${RESET}"
  elif (( $(echo "$num >= 0.55" | bc -l) )); then
    echo -e "${GREEN}$val${RESET}"
  elif (( $(echo "$num <= 0.30" | bc -l) )); then
    echo -e "${BOLD}${RED}$val${RESET}"
  elif (( $(echo "$num <= 0.45" | bc -l) )); then
    echo -e "${RED}$val${RESET}"
  else
    echo -e "${YELLOW}$val${RESET}"
  fi
}

color_result() {
  case "$1" in
    WIN)  echo -e "${BOLD}${GREEN}WIN${RESET}" ;;
    LOSS) echo -e "${BOLD}${RED}LOSS${RESET}" ;;
    ...)  echo -e "${YELLOW}...${RESET}" ;;
    *)    echo -e "${DIM}$1${RESET}" ;;
  esac
}

color_wr() {
  local n="$1"
  if [ -z "$n" ] || [ "$n" = "NULL" ]; then echo -e "${DIM}—${RESET}"; return; fi
  if (( $(echo "$n >= 55" | bc -l) )); then echo -e "${BOLD}${GREEN}${n}%${RESET}"
  elif (( $(echo "$n >= 48" | bc -l) )); then echo -e "${YELLOW}${n}%${RESET}"
  else echo -e "${RED}${n}%${RESET}"; fi
}

color_pnl() {
  local n="$1"
  if [ -z "$n" ] || [ "$n" = "NULL" ]; then echo -e "${DIM}\$0.00${RESET}"; return; fi
  if (( $(echo "$n >= 0" | bc -l) )); then echo -e "${BOLD}${GREEN}+\$${n}${RESET}"
  else echo -e "${BOLD}${RED}\$${n}${RESET}"; fi
}

color_regime() {
  case "$1" in
    trending_up)      echo -e "${GREEN}up${RESET}" ;;
    trending_down)    echo -e "${RED}down${RESET}" ;;
    high_uncertainty) echo -e "${YELLOW}hi_unc${RESET}" ;;
    ranging)          echo -e "${DIM}ranging${RESET}" ;;
    *)                echo -e "${DIM}${1}${RESET}" ;;
  esac
}

while true; do
  clear
  LATEST_LOG=$(ls -t "$LOG_DIR"/kronos_*.log 2>/dev/null | head -1)

  # ── Header ────────────────────────────────────────────────────────────
  echo -e "${BOLD}${BG_DARK}${WHITE}  ══════════════════════════════════════════════════════════  ${RESET}"
  echo -e "${BOLD}${BG_DARK}${CYAN}    KRONOS MONITOR  ${WHITE}—  $(date '+%H:%M:%S PDT')  —  ${DIM}refresh 30s      ${RESET}"
  echo -e "${BOLD}${BG_DARK}${WHITE}  ══════════════════════════════════════════════════════════  ${RESET}"

  # ── BG Loop ───────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ BG LOOP${RESET}  ${DIM}(last 3 candles)${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    grep "KronosBG:" "$LATEST_LOG" | tail -3 | while IFS= read -r line; do
      k5=$(echo "$line"    | grep -oE 'prob=[0-9.]+' | cut -d= -f2)
      k15=$(echo "$line"   | grep -oE 'prob_15min=[0-9.]+' | cut -d= -f2)
      candle=$(echo "$line" | grep -oE 'candle=[^ ]+' | sed 's/candle=//')
      k5_c=$(color_prob "$k5")
      k15_c=$(color_prob "$k15")
      echo -e "  ${DIM}$candle${RESET}  k5=${k5_c}  k15=${BOLD}${k15_c}${RESET}"
    done
  fi

  # ── Gate Rejections ───────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ GATE REJECTIONS${RESET}  ${DIM}(recent 10)${RESET}"
  echo -e "  ${DIM}time   gate  dir  fill   k15raw k15cal regime   result${RESET}"

  sqlite3 -separator '|' "$DB" "
    SELECT
      strftime('%H:%M', datetime(timestamp,'unixepoch','localtime')),
      'G'||failed_gate || CASE WHEN shadow=1 THEN '★' ELSE '' END,
      CASE direction WHEN 1 THEN 'YES' ELSE 'NO' END,
      would_be_fill_cents,
      ROUND(kronos_raw_15min, 2),
      ROUND(k15_calibrated_prob, 2),
      COALESCE(deepseek_regime, '?'),
      CASE WHEN outcome=1 THEN 'WIN'
           WHEN outcome=0 THEN 'LOSS'
           ELSE '...' END
    FROM gate_rejections
    ORDER BY timestamp DESC LIMIT 10;
  " 2>/dev/null | while IFS='|' read -r time gate dir fill k15raw k15cal regime result; do
    k15raw_c=$(color_prob "$k15raw")
    k15cal_c=$(color_prob "$k15cal")
    result_c=$(color_result "$result")
    regime_c=$(color_regime "$regime")

    fill_num=$(echo "$fill" | grep -oE '[0-9]+' | head -1)
    if [ -n "$fill_num" ] && (( fill_num <= 35 )) || (( fill_num >= 65 )); then
      fill_c="${BOLD}${MAGENTA}${fill}¢${RESET}"
    else
      fill_c="${WHITE}${fill}¢${RESET}"
    fi

    case "$dir" in
      YES) dir_c="${GREEN}YES${RESET}" ;;
      NO)  dir_c="${RED} NO${RESET}" ;;
    esac

    printf "  ${DIM}%-6s${RESET} %-6s %s  %-8s %-18s %-18s %-18s %s\n" \
      "$time" "$gate" "$dir_c" "$fill_c" "$k15raw_c" "$k15cal_c" "$regime_c" "$result_c"
  done

  # ── Trades ────────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ TRADES${RESET}  ${DIM}(recent 8)${RESET}"
  echo -e "  ${DIM}time   mkt    dir  fill   k15raw k15cal kelly      result${RESET}"

  sqlite3 -separator '|' "$DB" "
    SELECT
      strftime('%H:%M', datetime(substr(timestamp,1,19), 'localtime')),
      substr(ticker,17,2)||':'||substr(ticker,19,2),
      CASE direction WHEN 1 THEN 'YES' ELSE 'NO' END,
      fill_price_cents,
      ROUND(kronos_raw_15min, 2),
      ROUND(k15_calibrated_prob, 2),
      kelly_contracts||'x\$'||printf('%.2f', kelly_dollars),
      CASE WHEN outcome=1 THEN 'WIN'
           WHEN outcome=0 THEN 'LOSS'
           ELSE '...' END
    FROM trades
    ORDER BY timestamp DESC LIMIT 8;
  " 2>/dev/null | while IFS='|' read -r time mkt dir fill k15raw k15cal kelly result; do
    result_c=$(color_result "$result")
    k15raw_c=$(color_prob "$k15raw")
    k15cal_c=$(color_prob "$k15cal")
    case "$dir" in
      YES) dir_c="${GREEN}YES${RESET}" ;;
      NO)  dir_c="${RED} NO${RESET}" ;;
    esac
    printf "  ${DIM}%-6s${RESET} %-6s %s  ${WHITE}%-7s${RESET} %-18s %-18s ${YELLOW}%-12s${RESET} %s\n" \
      "$time" "$mkt" "$dir_c" "${fill}¢" "$k15raw_c" "$k15cal_c" "$kelly" "$result_c"
  done

  # ── P&L ───────────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ P&L${RESET}"

  today_stats=$(sqlite3 "$DB" "
    SELECT COUNT(*),
      SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),
      SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1.0 ELSE 0 END)/
        NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
      ROUND(SUM(CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
                     WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
                     ELSE 0 END), 2)
    FROM trades
    WHERE outcome IS NOT NULL
      AND date(datetime(substr(timestamp,1,19), 'localtime')) = date('now','localtime');
  " 2>/dev/null)
  t_total=$(echo "$today_stats" | cut -d'|' -f1)
  t_wins=$(echo "$today_stats"  | cut -d'|' -f2)
  t_losses=$(echo "$today_stats"| cut -d'|' -f3)
  t_wr=$(echo "$today_stats"    | cut -d'|' -f4)
  t_pnl=$(echo "$today_stats"   | cut -d'|' -f5)

  all_stats=$(sqlite3 "$DB" "
    SELECT COUNT(*),
      SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),
      SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1.0 ELSE 0 END)/
        NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
      ROUND(SUM(CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
                     WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
                     ELSE 0 END), 2)
    FROM trades WHERE outcome IS NOT NULL;
  " 2>/dev/null)
  total=$(echo "$all_stats"  | cut -d'|' -f1)
  wins=$(echo "$all_stats"   | cut -d'|' -f2)
  losses=$(echo "$all_stats" | cut -d'|' -f3)
  wr=$(echo "$all_stats"     | cut -d'|' -f4)
  pnl=$(echo "$all_stats"    | cut -d'|' -f5)

  t_wr_num=$(echo "$t_wr" | grep -oE '[0-9]+\.?[0-9]*' | head -1)
  t_pnl_num=$(echo "$t_pnl" | grep -oE '[-]?[0-9]+\.?[0-9]*' | head -1)
  wr_num=$(echo "$wr" | grep -oE '[0-9]+\.?[0-9]*' | head -1)
  pnl_num=$(echo "$pnl" | grep -oE '[-]?[0-9]+\.?[0-9]*' | head -1)

  echo -e "  ${BOLD}Today:${RESET}     Trades: ${WHITE}${t_total}${RESET}  W:${GREEN}${t_wins}${RESET} L:${RED}${t_losses}${RESET}  WR: $(color_wr "$t_wr_num")  Net: $(color_pnl "$t_pnl_num")"
  echo -e "  ${DIM}All-time:${RESET}  Trades: ${WHITE}${total}${RESET}  W:${GREEN}${wins}${RESET} L:${RED}${losses}${RESET}  WR: $(color_wr "$wr_num")  Net: $(color_pnl "$pnl_num")"

  # Gate rejection stats (resolved only, excluding shadow=1 Gate 7)
  echo -e "  ${DIM}Gate rejections (resolved, excl shadow):${RESET}"
  sqlite3 -separator '|' "$DB" "
    SELECT failed_gate,
      COUNT(*) as n,
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1.0 ELSE 0 END)/COUNT(*),1)
    FROM gate_rejections
    WHERE outcome IS NOT NULL AND shadow = 0
    GROUP BY failed_gate ORDER BY failed_gate;
  " 2>/dev/null | while IFS='|' read -r gate n wr_g; do
    wr_g_num=$(echo "$wr_g" | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    wr_g_c=$(color_wr "$wr_g_num")
    printf "    ${DIM}G%-3s${RESET} n=%-5s WR=%s\n" "$gate" "$n" "$wr_g_c"
  done

  # ── Calibrator ────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ CALIBRATOR${RESET}"
  if [ -f "$CAL_META" ]; then
    cal_rows=$(python3 -c "import json; d=json.load(open('$CAL_META')); print(d.get('total_rows_at_train','?'))" 2>/dev/null)
    cal_ts=$(python3 -c "import json; d=json.load(open('$CAL_META')); print(d.get('trained_at_timestamp','?')[:16])" 2>/dev/null)
    echo -e "  Last trained: ${WHITE}${cal_ts} UTC${RESET}  rows: ${WHITE}${cal_rows}${RESET}"
  fi
  # Compression spot-check (k15_raw → k15_cal for trending_up) from recent gate rejections
  compression=$(sqlite3 "$DB" "
    SELECT
      ROUND(MIN(kronos_raw_15min),2),
      ROUND(MAX(kronos_raw_15min),2),
      ROUND(MIN(k15_calibrated_prob),2),
      ROUND(MAX(k15_calibrated_prob),2),
      COUNT(*)
    FROM (
      SELECT kronos_raw_15min, k15_calibrated_prob FROM trades
      WHERE k15_calibrated_prob IS NOT NULL AND kronos_raw_15min IS NOT NULL
        AND kronos_raw_15min >= 0.70
      UNION ALL
      SELECT kronos_raw_15min, k15_calibrated_prob FROM gate_rejections
      WHERE k15_calibrated_prob IS NOT NULL AND kronos_raw_15min IS NOT NULL
        AND kronos_raw_15min >= 0.70
    ) ORDER BY rowid DESC LIMIT 50;
  " 2>/dev/null)
  if [ -n "$compression" ]; then
    cr_min=$(echo "$compression" | cut -d'|' -f1)
    cr_max=$(echo "$compression" | cut -d'|' -f2)
    cc_min=$(echo "$compression" | cut -d'|' -f3)
    cc_max=$(echo "$compression" | cut -d'|' -f4)
    cc_n=$(echo "$compression"   | cut -d'|' -f5)
    echo -e "  k15_raw≥0.70 range: ${WHITE}${cr_min}–${cr_max}${RESET} → k15_cal: ${YELLOW}${cc_min}–${cc_max}${RESET}  ${DIM}(n=${cc_n} strong signals)${RESET}"
  fi

  # ── Candle Logger / Regime v2 ─────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ REGIME V2 DATA${RESET}  ${DIM}(candle_features)${RESET}"
  candle_stats=$(sqlite3 "$DB" "
    SELECT COUNT(*),
      strftime('%m-%d %H:%M', MAX(candle_ts)),
      ROUND(COUNT(*) * 1.0 / 96, 1)
    FROM candle_features;
  " 2>/dev/null)
  cf_count=$(echo "$candle_stats" | cut -d'|' -f1)
  cf_last=$(echo "$candle_stats"  | cut -d'|' -f2)
  cf_days=$(echo "$candle_stats"  | cut -d'|' -f3)
  cf_needed=672
  if [ -n "$cf_count" ] && [ "$cf_count" -gt 0 ]; then
    cf_remain=$(( cf_needed - cf_count ))
    cf_eta_days=$(echo "scale=1; $cf_remain / 96" | bc)
    if (( $(echo "$cf_remain <= 0" | bc -l) )); then
      cf_status="${BOLD}${GREEN}READY${RESET}"
    else
      cf_status="${YELLOW}${cf_remain} more needed (~${cf_eta_days} days)${RESET}"
    fi
    echo -e "  Candles: ${WHITE}${cf_count}${RESET} / ${cf_needed}  Days: ${WHITE}${cf_days}${RESET}  Last: ${DIM}${cf_last} UTC${RESET}"
    echo -e "  Status: ${cf_status}"
  else
    echo -e "  ${DIM}No candles logged yet${RESET}"
  fi

  # ── Regime ────────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ REGIME${RESET}  ${DIM}(latest)${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    regime_line=$(grep "regime:features" "$LATEST_LOG" | tail -1)
    cvd=$(echo "$regime_line"  | grep -oE "'cvd_normalized': [-0-9.]+" | grep -oE '[-0-9.]+$')
    lp=$(echo "$regime_line"   | grep -oE "'large_print_direction': [-0-9.]+" | grep -oE '[-0-9.]+$')
    fund=$(echo "$regime_line" | grep -oE "'funding_rate': [-0-9.]+" | grep -oE '[-0-9.]+$')
    fg=$(echo "$regime_line"   | grep -oE "'fear_greed_label': '[^']+'" | grep -oE "'[^']+'$" | tr -d "'")

    cvd_r=$(echo "$cvd" | grep -oE '[-]?[0-9.]+')
    if [ -n "$cvd_r" ] && (( $(echo "$cvd_r >= 0.3" | bc -l) )); then
      cvd_c="${BOLD}${GREEN}CVD:${cvd}${RESET}"
    elif [ -n "$cvd_r" ] && (( $(echo "$cvd_r <= -0.3" | bc -l) )); then
      cvd_c="${BOLD}${RED}CVD:${cvd}${RESET}"
    else
      cvd_c="${YELLOW}CVD:${cvd}${RESET}"
    fi

    lp_r=$(echo "$lp" | grep -oE '[-]?[0-9.]+')
    if [ -n "$lp_r" ] && (( $(echo "$lp_r >= 0.3" | bc -l) )); then
      lp_c="${GREEN}LP:${lp}${RESET}"
    elif [ -n "$lp_r" ] && (( $(echo "$lp_r <= -0.3" | bc -l) )); then
      lp_c="${RED}LP:${lp}${RESET}"
    else
      lp_c="${YELLOW}LP:${lp}${RESET}"
    fi

    echo -e "  ${cvd_c}   ${lp_c}   ${DIM}fund:${fund}   fear/greed: ${fg}${RESET}"
  fi

  # ── Last activity ─────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ LAST ACTIVITY${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    grep -E "PAPER|resolved|failed|KronosBG|Circuit breaker" "$LATEST_LOG" | tail -4 | while IFS= read -r line; do
      if echo "$line" | grep -q "PAPER"; then
        echo -e "  ${BOLD}${GREEN}$(echo "$line" | sed 's/.*PAPER\] /[PAPER] /')${RESET}"
      elif echo "$line" | grep -q "WIN"; then
        echo -e "  ${GREEN}$(echo "$line" | sed 's/.*_resolutions:[0-9]*[[:space:]]*//')${RESET}"
      elif echo "$line" | grep -q "LOSS"; then
        echo -e "  ${RED}$(echo "$line" | sed 's/.*_resolutions:[0-9]*[[:space:]]*//')${RESET}"
      elif echo "$line" | grep -q "Circuit breaker"; then
        echo -e "  ${BOLD}${RED}$(echo "$line" | sed 's/.*\] //')${RESET}"
      else
        echo -e "  ${DIM}$(echo "$line" | sed 's/.*\] //')${RESET}"
      fi
    done
  fi

  echo ""
  echo -e "${DIM}  ──────────────────────────────────────────────────────────${RESET}"
  echo -e "${DIM}  Next refresh in 30s  (Ctrl+C to stop)${RESET}"
  sleep 30
done
