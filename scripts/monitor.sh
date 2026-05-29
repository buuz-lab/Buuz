#!/usr/bin/env bash
# Kronos live monitor — run in a separate terminal tab
# Refreshes every 30 seconds. Ctrl+C to stop.

DB="/Users/ezrakornberg/Kronos V2/trades.db"
LOG_DIR="/Users/ezrakornberg/Kronos V2/logs"

# ── Colors ────────────────────────────────────────────────────────────────────
RESET='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

BLACK='\033[30m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
MAGENTA='\033[35m'
CYAN='\033[36m'
WHITE='\033[37m'

BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'
BG_DARK='\033[40m'

# Color a probability value: green=bullish, red=bearish, yellow=neutral
color_prob() {
  local val="$1"
  local num
  num=$(echo "$val" | grep -oE '[0-9]+\.?[0-9]*' | head -1)
  if [ -z "$num" ]; then echo -e "${DIM}$val${RESET}"; return; fi
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

color_cvd() {
  local val
  val=$(echo "$1" | grep -oE '[-]?[0-9]+\.?[0-9]*' | head -1)
  if [ -z "$val" ]; then echo "$1"; return; fi
  if (( $(echo "$val >= 0.3" | bc -l) )); then
    echo -e "${GREEN}CVD:$val${RESET}"
  elif (( $(echo "$val <= -0.3" | bc -l) )); then
    echo -e "${RED}CVD:$val${RESET}"
  else
    echo -e "${YELLOW}CVD:$val${RESET}"
  fi
}

while true; do
  clear
  LATEST_LOG=$(ls -t "$LOG_DIR"/kronos_*.log 2>/dev/null | head -1)

  # ── Header ────────────────────────────────────────────────────────
  echo -e "${BOLD}${BG_DARK}${WHITE}  ══════════════════════════════════════════════════════════  ${RESET}"
  echo -e "${BOLD}${BG_DARK}${CYAN}    KRONOS MONITOR  ${WHITE}—  $(date '+%H:%M:%S PDT')  —  ${DIM}refresh 30s      ${RESET}"
  echo -e "${BOLD}${BG_DARK}${WHITE}  ══════════════════════════════════════════════════════════  ${RESET}"

  # ── BG Loop ───────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ BG LOOP${RESET}  ${DIM}(last 3 candles)${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    grep "KronosBG:" "$LATEST_LOG" | tail -3 | while IFS= read -r line; do
      # Extract fields
      k5=$(echo "$line"   | grep -oE 'prob=[0-9.]+' | grep -oE '[0-9.]+')
      k15=$(echo "$line"  | grep -oE 'prob_15min=[0-9.]+' | grep -oE '[0-9.]+')
      candle=$(echo "$line" | grep -oE 'candle=[^ ]+' | sed 's/candle=//')
      k5_c=$(color_prob "$k5")
      k15_c=$(color_prob "$k15")
      echo -e "  ${DIM}$candle${RESET}  k5=${k5_c}  k15=${BOLD}${k15_c}${RESET}"
    done
  fi

  # ── Gate Rejections ───────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ GATE REJECTIONS${RESET}  ${DIM}(recent 8)${RESET}"
  echo -e "  ${DIM}time   slot   k15    k15cal fill  regime            result${RESET}"

  sqlite3 -separator '|' "$DB" "
    SELECT
      strftime('%H:%M', datetime(timestamp,'unixepoch')),
      CASE WHEN candle_progress < 0.15 THEN 't=0'
           WHEN candle_progress < 0.55 THEN 't+5'
           ELSE 't+10' END,
      ROUND(kronos_raw_15min,2),
      ROUND(k15_calibrated_prob,2),
      would_be_fill_cents,
      deepseek_regime,
      CASE WHEN outcome=1 THEN 'WIN'
           WHEN outcome=0 THEN 'LOSS'
           ELSE '...' END
    FROM gate_rejections
    WHERE k15_calibrated_prob IS NOT NULL
    ORDER BY timestamp DESC LIMIT 8;
  " 2>/dev/null | while IFS='|' read -r time slot k15 k15cal fill regime result; do
    k15_c=$(color_prob "$k15")
    k15cal_c=$(color_prob "$k15cal")
    result_c=$(color_result "$result")

    # Color fill: cheap (≤35¢) = bright edge opportunity
    fill_num=$(echo "$fill" | grep -oE '[0-9]+' | head -1)
    if [ -n "$fill_num" ] && (( fill_num <= 35 )); then
      fill_c="${BOLD}${MAGENTA}${fill}¢${RESET}"
    elif [ -n "$fill_num" ] && (( fill_num >= 65 )); then
      fill_c="${BOLD}${MAGENTA}${fill}¢${RESET}"
    else
      fill_c="${WHITE}${fill}¢${RESET}"
    fi

    # Color regime
    case "$regime" in
      trending_up)   regime_c="${GREEN}$regime${RESET}" ;;
      trending_down) regime_c="${RED}$regime${RESET}" ;;
      high_uncertainty) regime_c="${YELLOW}$regime${RESET}" ;;
      ranging)       regime_c="${DIM}$regime${RESET}" ;;
      *)             regime_c="${DIM}$regime${RESET}" ;;
    esac

    printf "  ${DIM}%-6s${RESET} ${DIM}%-6s${RESET} %-18s %-18s %-14s %-28s %s\n" \
      "$time" "$slot" "$k15_c" "$k15cal_c" "$fill_c" "$regime_c" "$result_c"
  done

  # ── Trades ────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ TRADES${RESET}  ${DIM}(recent 6)${RESET}"
  echo -e "  ${DIM}time   market      dir   fill   kelly   result${RESET}"

  sqlite3 -separator '|' "$DB" "
    SELECT
      strftime('%H:%M', datetime(timestamp,'unixepoch')),
      replace(ticker,'KXBTC15M-26MAY27',''),
      CASE direction WHEN 1 THEN 'YES' ELSE 'NO' END,
      fill_price_cents,
      printf('%.2f', kelly_dollars),
      CASE WHEN outcome=1 THEN 'WIN'
           WHEN outcome=0 THEN 'LOSS'
           ELSE '...' END
    FROM trades
    ORDER BY timestamp DESC LIMIT 6;
  " 2>/dev/null | while IFS='|' read -r time market dir fill kelly result; do
    result_c=$(color_result "$result")
    case "$dir" in
      YES) dir_c="${GREEN}YES${RESET}" ;;
      NO)  dir_c="${RED} NO${RESET}" ;;
    esac
    fill_c="${WHITE}${fill}¢${RESET}"
    printf "  ${DIM}%-6s${RESET} %-12s %s  %-8s ${YELLOW}\$%-7s${RESET} %s\n" \
      "$time" "$market" "$dir_c" "$fill_c" "$kelly" "$result_c"
  done

  # ── P&L ───────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ P&L${RESET}"

  # timestamps stored as UTC ISO8601; PDT = UTC-7
  # strftime('%Y-%m-%d', substr(timestamp,1,19), '-7 hours') converts to PDT date
  today_stats=$(sqlite3 "$DB" "
    SELECT
      COUNT(*),
      SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),
      SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END)/
        NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
      ROUND(SUM(
        CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
             WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
             ELSE 0 END), 2)
    FROM trades
    WHERE outcome IS NOT NULL
      AND strftime('%Y-%m-%d', substr(timestamp,1,19), '-7 hours')
        = strftime('%Y-%m-%d', 'now', '-7 hours');
  " 2>/dev/null)
  t_total=$(echo "$today_stats" | cut -d'|' -f1)
  t_wins=$(echo "$today_stats"  | cut -d'|' -f2)
  t_losses=$(echo "$today_stats"| cut -d'|' -f3)
  t_wr=$(echo "$today_stats"    | cut -d'|' -f4)
  t_pnl=$(echo "$today_stats"   | cut -d'|' -f5)

  all_stats=$(sqlite3 "$DB" "
    SELECT
      COUNT(*),
      SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),
      SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),
      ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END)/
        NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
      ROUND(SUM(
        CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
             WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
             ELSE 0 END), 2)
    FROM trades WHERE outcome IS NOT NULL;
  " 2>/dev/null)
  total=$(echo "$all_stats"  | cut -d'|' -f1)
  wins=$(echo "$all_stats"   | cut -d'|' -f2)
  losses=$(echo "$all_stats" | cut -d'|' -f3)
  wr=$(echo "$all_stats"     | cut -d'|' -f4)
  pnl=$(echo "$all_stats"    | cut -d'|' -f5)

  # color helpers
  _color_wr() {
    local n="$1" val="$2"
    if [ -n "$n" ] && (( $(echo "$n >= 55" | bc -l) )); then echo -e "${BOLD}${GREEN}${val}%${RESET}"
    elif [ -n "$n" ] && (( $(echo "$n >= 48" | bc -l) )); then echo -e "${YELLOW}${val}%${RESET}"
    else echo -e "${RED}${val}%${RESET}"; fi
  }
  _color_pnl() {
    local n="$1" val="$2"
    if [ -n "$n" ] && (( $(echo "$n >= 0" | bc -l) )); then echo -e "${BOLD}${GREEN}+\$${val}${RESET}"
    else echo -e "${BOLD}${RED}\$${val}${RESET}"; fi
  }

  t_wr_num=$(echo "$t_wr"   | grep -oE '[0-9]+\.?[0-9]*' | head -1)
  t_pnl_num=$(echo "$t_pnl" | grep -oE '[-]?[0-9]+\.?[0-9]*' | head -1)
  wr_num=$(echo "$wr"       | grep -oE '[0-9]+\.?[0-9]*' | head -1)
  pnl_num=$(echo "$pnl"     | grep -oE '[-]?[0-9]+\.?[0-9]*' | head -1)

  t_wr_c=$(_color_wr "$t_wr_num" "$t_wr")
  t_pnl_c=$(_color_pnl "$t_pnl_num" "$t_pnl")
  wr_c=$(_color_wr "$wr_num" "$wr")
  pnl_c=$(_color_pnl "$pnl_num" "$pnl")

  echo -e "  ${BOLD}Today (PDT):${RESET}  Trades: ${WHITE}${t_total}${RESET}  Wins: ${GREEN}${t_wins}${RESET}  Losses: ${RED}${t_losses}${RESET}  WR: ${t_wr_c}  Net: ${t_pnl_c}"
  echo -e "  ${DIM}All-time:${RESET}     Trades: ${WHITE}${total}${RESET}  Wins: ${GREEN}${wins}${RESET}  Losses: ${RED}${losses}${RESET}  WR: ${wr_c}  Net: ${pnl_c}"

  # ── Regime ────────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ REGIME${RESET}  ${DIM}(latest)${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    regime_line=$(grep "regime:features" "$LATEST_LOG" | tail -1)
    cvd=$(echo "$regime_line"   | grep -oE "'cvd_normalized': [-0-9.]+" | grep -oE '[-0-9.]+$')
    lp=$(echo "$regime_line"    | grep -oE "'large_print_direction': [-0-9.]+" | grep -oE '[-0-9.]+$')
    fund=$(echo "$regime_line"  | grep -oE "'funding_rate': [-0-9.]+" | grep -oE '[-0-9.]+$')
    fg=$(echo "$regime_line"    | grep -oE "'fear_greed_label': '[^']+'" | grep -oE "'[^']+'$" | tr -d "'")

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

  # ── Last activity ─────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}${CYAN}▶ LAST ACTIVITY${RESET}"
  if [ -n "$LATEST_LOG" ]; then
    grep -E "PAPER|resolved|failed|KronosBG" "$LATEST_LOG" | tail -3 | while IFS= read -r line; do
      if echo "$line" | grep -q "PAPER"; then
        echo -e "  ${BOLD}${GREEN}$(echo "$line" | sed 's/.*PAPER\] /[PAPER] /')${RESET}"
      elif echo "$line" | grep -q "WIN"; then
        echo -e "  ${GREEN}$(echo "$line" | sed 's/.*_resolutions:1[0-9]*[[:space:]]*//')${RESET}"
      elif echo "$line" | grep -q "LOSS"; then
        echo -e "  ${RED}$(echo "$line" | sed 's/.*_resolutions:1[0-9]*[[:space:]]*//')${RESET}"
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
