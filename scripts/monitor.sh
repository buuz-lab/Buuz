#!/usr/bin/env bash
# Kronos live monitor — run in a separate terminal tab
# Refreshes every 30 seconds. Ctrl+C to stop.

DB="/Users/ezrakornberg/Kronos V2/trades.db"
LOG_DIR="/Users/ezrakornberg/Kronos V2/logs"
CAL_META="/Users/ezrakornberg/Kronos V2/models/calibrator_last_trained.json"

# ── Colors ────────────────────────────────────────────────────────────────────
RESET='\033[0m'; BOLD='\033[1m'; DIM='\033[2m'
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'
CYAN='\033[36m'; WHITE='\033[37m'; MAGENTA='\033[35m'
BG_DARK='\033[40m'

# Pre-pad a value to a fixed width, then apply a color based on numeric comparison.
# Usage: cpad_prob <raw_number> <display_width>
# Reads the value from $1, pads to $2, returns colored string.
cpad_prob() {
  local val="$1" w="${2:-6}"
  local num; num=$(echo "$val" | grep -oE '^[0-9]*\.?[0-9]*$' | head -1)
  local disp; disp=$(printf "%-${w}s" "${val:-N/A}")
  if [ -z "$num" ]; then echo -e "${DIM}${disp}${RESET}"; return; fi
  if   (( $(echo "$num >= 0.70" | bc -l) )); then echo -e "${BOLD}${GREEN}${disp}${RESET}"
  elif (( $(echo "$num >= 0.55" | bc -l) )); then echo -e "${GREEN}${disp}${RESET}"
  elif (( $(echo "$num <= 0.30" | bc -l) )); then echo -e "${BOLD}${RED}${disp}${RESET}"
  elif (( $(echo "$num <= 0.45" | bc -l) )); then echo -e "${RED}${disp}${RESET}"
  else echo -e "${YELLOW}${disp}${RESET}"; fi
}

cpad_result() {
  local val="$1" w="${2:-6}"
  local disp; disp=$(printf "%-${w}s" "$val")
  case "$val" in
    WIN)  echo -e "${BOLD}${GREEN}${disp}${RESET}" ;;
    LOSS) echo -e "${BOLD}${RED}${disp}${RESET}" ;;
    ...)  echo -e "${YELLOW}${disp}${RESET}" ;;
    *)    echo -e "${DIM}${disp}${RESET}" ;;
  esac
}

cpad_regime() {
  local val="$1" w="${2:-8}"
  local short
  case "$val" in
    trending_up)      short="up" ;;
    trending_down)    short="down" ;;
    high_uncertainty) short="hi_unc" ;;
    ranging)          short="ranging" ;;
    *)                short="${val:0:8}" ;;
  esac
  local disp; disp=$(printf "%-${w}s" "$short")
  case "$val" in
    trending_up)      echo -e "${GREEN}${disp}${RESET}" ;;
    trending_down)    echo -e "${RED}${disp}${RESET}" ;;
    high_uncertainty) echo -e "${YELLOW}${disp}${RESET}" ;;
    ranging)          echo -e "${DIM}${disp}${RESET}" ;;
    *)                echo -e "${DIM}${disp}${RESET}" ;;
  esac
}

cpad_dir() {
  local val="$1"
  case "$val" in
    YES) echo -e "${GREEN}YES${RESET}" ;;
    NO)  echo -e "${RED} NO${RESET}" ;;
    *)   echo -e "${DIM} ??${RESET}" ;;
  esac
}

cpad_wr() {
  local n="$1" w="${2:-6}"
  local disp; disp=$(printf "%-${w}s" "${n}%")
  if [ -z "$n" ] || [ "$n" = "NULL" ]; then echo -e "${DIM}$(printf "%-${w}s" "—")${RESET}"; return; fi
  if   (( $(echo "$n >= 55" | bc -l) )); then echo -e "${BOLD}${GREEN}${disp}${RESET}"
  elif (( $(echo "$n >= 48" | bc -l) )); then echo -e "${YELLOW}${disp}${RESET}"
  else echo -e "${RED}${disp}${RESET}"; fi
}

cpad_pnl() {
  local n="$1"
  if [ -z "$n" ] || [ "$n" = "NULL" ]; then echo -e "${DIM}pending ${RESET}"; return; fi
  if (( $(echo "$n >= 0" | bc -l) )); then echo -e "${BOLD}${GREEN}+\$${n}${RESET}"
  else echo -e "${BOLD}${RED}\$${n}${RESET}"; fi
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
      k5=$(echo "$line"  | grep -oE 'prob=[0-9.]+' | cut -d= -f2)
      k15=$(echo "$line" | grep -oE 'prob_15min=[0-9.]+' | cut -d= -f2)
      candle=$(echo "$line" | grep -oE 'candle=[^ ]+' | sed 's/candle=//')
      echo -e "  ${DIM}${candle}${RESET}  k5=$(cpad_prob "$k5" 6)  k15=$(cpad_prob "$k15" 6)"
    done
  fi

  # ── Gate Rejections ───────────────────────────────────────────────────
  # Columns: time(5) gate(5) dir(3) fill(5) k15raw(6) k15cal(6) regime(8) result(5)
  echo ""
  echo -e "${BOLD}${CYAN}▶ GATE REJECTIONS${RESET}  ${DIM}(recent 10)${RESET}"
  printf "  ${DIM}%-5s  %-5s  %-3s  %-5s  %-6s  %-6s  %-8s  %-5s${RESET}\n" \
    "time" "gate" "dir" "fill" "k15raw" "k15cal" "regime" "result"

  sqlite3 -separator '|' "$DB" "
    SELECT
      strftime('%H:%M', datetime(timestamp,'unixepoch','localtime')),
      'G'||failed_gate || CASE WHEN shadow=1 THEN '*' ELSE '' END,
      CASE direction WHEN 1 THEN 'YES' ELSE 'NO' END,
      COALESCE(would_be_fill_cents,'?'),
      COALESCE(ROUND(kronos_raw_15min,2),'N/A'),
      COALESCE(ROUND(k15_calibrated_prob,2),'N/A'),
      COALESCE(deepseek_regime,'?'),
      CASE WHEN outcome=1 THEN 'WIN'
           WHEN outcome=0 THEN 'LOSS'
           ELSE '...' END
    FROM gate_rejections
    ORDER BY timestamp DESC LIMIT 10;
  " 2>/dev/null | while IFS='|' read -r time gate dir fill k15raw k15cal regime result; do
    p_time=$(printf "%-5s" "$time")
    p_gate=$(printf "%-5s" "$gate")
    p_fill=$(printf "%-5s" "${fill}¢")

    # color fill: extreme prices are notable
    fill_num=$(echo "$fill" | grep -oE '^[0-9]+$')
    if [ -n "$fill_num" ] && { (( fill_num <= 35 )) || (( fill_num >= 65 )); }; then
      c_fill="${MAGENTA}${p_fill}${RESET}"
    else
      c_fill="${WHITE}${p_fill}${RESET}"
    fi

    echo -e "  ${DIM}${p_time}${RESET}  ${p_gate}  $(cpad_dir "$dir")  ${c_fill}  $(cpad_prob "$k15raw" 6)  $(cpad_prob "$k15cal" 6)  $(cpad_regime "$regime" 8)  $(cpad_result "$result" 5)"
  done

  # ── Trades ────────────────────────────────────────────────────────────
  # trades.timestamp is ISO8601 UTC string; gate_rejections uses Unix epoch
  # Columns: time(5) mkt(5) dir(3) fill(5) k15raw(6) k15cal(6) kelly(8) result(5) pnl(8)
  echo ""
  echo -e "${BOLD}${CYAN}▶ TRADES${RESET}  ${DIM}(recent 8)${RESET}"
  printf "  ${DIM}%-5s  %-5s  %-3s  %-5s  %-6s  %-6s  %-8s  %-5s  %-8s${RESET}\n" \
    "time" "mkt" "dir" "fill" "k15raw" "k15cal" "kelly" "result" "p&l"

  sqlite3 -separator '|' "$DB" "
    SELECT
      strftime('%H:%M', datetime(substr(timestamp,1,19), 'localtime')),
      substr(ticker,17,2)||':'||substr(ticker,19,2),
      CASE direction WHEN 1 THEN 'YES' ELSE 'NO' END,
      fill_price_cents,
      COALESCE(ROUND(kronos_raw_15min,2),'N/A'),
      COALESCE(ROUND(k15_calibrated_prob,2),'N/A'),
      kelly_contracts||'x',
      CASE WHEN outcome=1 THEN 'WIN'
           WHEN outcome=0 THEN 'LOSS'
           ELSE '...' END,
      COALESCE(ROUND(pnl_dollars,2),'')
    FROM trades
    ORDER BY timestamp DESC LIMIT 8;
  " 2>/dev/null | while IFS='|' read -r time mkt dir fill k15raw k15cal kelly result pnl; do
    p_time=$(printf "%-5s" "$time")
    p_mkt=$(printf "%-5s" "$mkt")
    p_fill=$(printf "%-5s" "${fill}¢")
    p_kelly=$(printf "%-8s" "$kelly")

    echo -e "  ${DIM}${p_time}${RESET}  ${p_mkt}  $(cpad_dir "$dir")  ${WHITE}${p_fill}${RESET}  $(cpad_prob "$k15raw" 6)  $(cpad_prob "$k15cal" 6)  ${YELLOW}${p_kelly}${RESET}  $(cpad_result "$result" 5)  $(cpad_pnl "$pnl")"
  done

  # ── P&L ───────────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ P&L${RESET}"

  today_stats=$(sqlite3 "$DB" "
    SELECT COUNT(*),
      COALESCE(SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),0),
      COALESCE(SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),0),
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1.0 ELSE 0 END)/
        NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
      ROUND(SUM(CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
                     WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
                     ELSE 0 END),2)
    FROM trades
    WHERE outcome IS NOT NULL
      AND date(datetime(substr(timestamp,1,19),'localtime')) = date('now','localtime');
  " 2>/dev/null)

  all_stats=$(sqlite3 "$DB" "
    SELECT COUNT(*),
      COALESCE(SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),0),
      COALESCE(SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),0),
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1.0 ELSE 0 END)/
        NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
      ROUND(SUM(CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
                     WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
                     ELSE 0 END),2)
    FROM trades WHERE outcome IS NOT NULL;
  " 2>/dev/null)

  IFS='|' read -r t_n t_w t_l t_wr t_pnl <<< "$today_stats"
  IFS='|' read -r a_n a_w a_l a_wr a_pnl <<< "$all_stats"
  t_wr_n=$(echo "$t_wr" | grep -oE '[0-9]+\.?[0-9]*')
  a_wr_n=$(echo "$a_wr" | grep -oE '[0-9]+\.?[0-9]*')
  t_pnl_n=$(echo "$t_pnl" | grep -oE '[-]?[0-9]+\.?[0-9]*')
  a_pnl_n=$(echo "$a_pnl" | grep -oE '[-]?[0-9]+\.?[0-9]*')

  printf "  ${BOLD}%-10s${RESET}  trades:${WHITE}%4s${RESET}  W:${GREEN}%3s${RESET}  L:${RED}%3s${RESET}  WR:%s  net:%s\n" \
    "Today" "$t_n" "$t_w" "$t_l" "$(cpad_wr "$t_wr_n" 7)" "$(cpad_pnl "$t_pnl_n")"
  printf "  ${DIM}%-10s${RESET}  trades:${WHITE}%4s${RESET}  W:${GREEN}%3s${RESET}  L:${RED}%3s${RESET}  WR:%s  net:%s\n" \
    "All-time" "$a_n" "$a_w" "$a_l" "$(cpad_wr "$a_wr_n" 7)" "$(cpad_pnl "$a_pnl_n")"

  # Gate breakdown
  echo ""
  printf "  ${DIM}%-5s  %6s  %7s${RESET}\n" "gate" "n" "WR"
  sqlite3 -separator '|' "$DB" "
    SELECT failed_gate, COUNT(*),
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1.0 ELSE 0 END)/COUNT(*),1)
    FROM gate_rejections
    WHERE outcome IS NOT NULL AND shadow = 0
    GROUP BY failed_gate ORDER BY failed_gate;
  " 2>/dev/null | while IFS='|' read -r gate n wr_g; do
    wr_n=$(echo "$wr_g" | grep -oE '[0-9]+\.?[0-9]*')
    printf "  ${DIM}G%-4s${RESET}  %6s  %s\n" "$gate" "$n" "$(cpad_wr "$wr_n" 7)"
  done

  # ── Calibrator ────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ CALIBRATOR${RESET}"
  if [ -f "$CAL_META" ]; then
    cal_rows=$(python3 -c "import json; d=json.load(open('$CAL_META')); print(d.get('total_rows_at_train','?'))" 2>/dev/null)
    cal_ts=$(python3 -c "import json; d=json.load(open('$CAL_META')); print(d.get('trained_at_timestamp','?')[:16])" 2>/dev/null)
    printf "  last trained: ${WHITE}%-19s UTC${RESET}  rows: ${WHITE}%s${RESET}\n" "$cal_ts" "$cal_rows"
  fi
  # Compression spot from recent strong signals
  comp=$(sqlite3 "$DB" "
    SELECT ROUND(AVG(kronos_raw_15min),2), ROUND(AVG(k15_calibrated_prob),2), COUNT(*)
    FROM (
      SELECT kronos_raw_15min, k15_calibrated_prob FROM trades
        WHERE k15_calibrated_prob IS NOT NULL AND kronos_raw_15min >= 0.70
      UNION ALL
      SELECT kronos_raw_15min, k15_calibrated_prob FROM gate_rejections
        WHERE k15_calibrated_prob IS NOT NULL AND kronos_raw_15min >= 0.70
    ) LIMIT 100;
  " 2>/dev/null)
  IFS='|' read -r comp_raw comp_cal comp_n <<< "$comp"
  if [ -n "$comp_n" ] && [ "$comp_n" -gt 0 ] 2>/dev/null; then
    printf "  k15_raw≥0.70 avg: ${WHITE}%.2f${RESET} → k15_cal avg: ${YELLOW}%.2f${RESET}  ${DIM}(n=%s strong signals)${RESET}\n" \
      "${comp_raw:-0}" "${comp_cal:-0}" "$comp_n"
  fi

  # ── Candle Logger / Regime v2 ─────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ REGIME V2 DATA${RESET}  ${DIM}(candle_features)${RESET}"
  IFS='|' read -r cf_n cf_last cf_days < <(sqlite3 "$DB" "
    SELECT COUNT(*), strftime('%m-%d %H:%M', MAX(candle_ts)), ROUND(COUNT(*)/96.0,1)
    FROM candle_features;" 2>/dev/null)
  cf_needed=672
  if [ -n "$cf_n" ] && [ "$cf_n" -gt 0 ] 2>/dev/null; then
    cf_remain=$(( cf_needed - cf_n ))
    cf_eta=$(echo "scale=1; $cf_remain / 96" | bc)
    if (( cf_remain <= 0 )); then
      cf_status="${BOLD}${GREEN}READY TO RETRAIN${RESET}"
    else
      cf_status="${YELLOW}${cf_remain} more needed  (~${cf_eta} days)${RESET}"
    fi
    printf "  candles: ${WHITE}%4s${RESET} / %s  days: ${WHITE}%s${RESET}  last: ${DIM}%s UTC${RESET}\n" \
      "$cf_n" "$cf_needed" "$cf_days" "$cf_last"
    echo -e "  status:  ${cf_status}"
  else
    echo -e "  ${DIM}no candles logged yet${RESET}"
  fi

  # ── Regime ────────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ REGIME${RESET}  ${DIM}(latest)${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    rl=$(grep "regime:features" "$LATEST_LOG" | tail -1)
    cvd=$(echo "$rl" | grep -oE "'cvd_normalized': [-0-9.]+" | grep -oE '[-0-9.]+$')
    lp=$(echo "$rl"  | grep -oE "'large_print_direction': [-0-9.]+" | grep -oE '[-0-9.]+$')
    fund=$(echo "$rl"| grep -oE "'funding_rate': [-0-9.]+" | grep -oE '[-0-9.]+$')
    fg=$(echo "$rl"  | grep -oE "'fear_greed_label': '[^']+'" | grep -oE "'[^']+'$" | tr -d "'")

    cvd_v=$(echo "$cvd" | grep -oE '[-]?[0-9.]+')
    lp_v=$(echo "$lp"   | grep -oE '[-]?[0-9.]+')
    cvd_r=$(printf "%.3f" "${cvd_v:-0}" 2>/dev/null)
    lp_r=$(printf  "%.3f" "${lp_v:-0}"  2>/dev/null)
    fund_r=$(printf "%.6f" "$(echo "$fund" | grep -oE '[-]?[0-9.]+')" 2>/dev/null)
    [ -n "$cvd_v" ] && (( $(echo "$cvd_v >= 0.3"  | bc -l) )) && cvd_c="${BOLD}${GREEN}" || \
    { [ -n "$cvd_v" ] && (( $(echo "$cvd_v <= -0.3" | bc -l) )) && cvd_c="${BOLD}${RED}" || cvd_c="${YELLOW}"; }
    [ -n "$lp_v"  ] && (( $(echo "$lp_v >= 0.3"  | bc -l) )) && lp_c="${GREEN}" || \
    { [ -n "$lp_v"  ] && (( $(echo "$lp_v <= -0.3" | bc -l) )) && lp_c="${RED}" || lp_c="${YELLOW}"; }

    printf "  ${cvd_c}CVD:%-8s${RESET}  ${lp_c}LP:%-8s${RESET}  ${DIM}fund:%-10s  %s${RESET}\n" \
      "$cvd_r" "$lp_r" "$fund_r" "$fg"
  fi

  # ── Last activity ─────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ LAST ACTIVITY${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    grep -E "PAPER|resolved|KronosBG|Circuit breaker" "$LATEST_LOG" | tail -4 | while IFS= read -r line; do
      msg=$(echo "$line" | sed 's/.*\] //')
      if echo "$line" | grep -q "PAPER"; then
        echo -e "  ${BOLD}${GREEN}${msg}${RESET}"
      elif echo "$line" | grep -qE "WIN|would_have=WON"; then
        echo -e "  ${GREEN}${msg}${RESET}"
      elif echo "$line" | grep -qE "LOSS|would_have=LOST"; then
        echo -e "  ${RED}${msg}${RESET}"
      elif echo "$line" | grep -q "Circuit breaker tripped"; then
        echo -e "  ${BOLD}${RED}${msg}${RESET}"
      else
        echo -e "  ${DIM}${msg}${RESET}"
      fi
    done
  fi

  echo ""
  echo -e "${DIM}  ──────────────────────────────────────────────────────────${RESET}"
  echo -e "${DIM}  Next refresh in 30s  (Ctrl+C to stop)${RESET}"
  sleep 30
done
