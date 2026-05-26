#!/bin/bash
# Wrapper for launchd: loads .env then starts Kronos.
# launchd does not source shell profiles, so PATH and secrets must be set here.

cd "/Users/ezrakornberg/Kronos V2"

set -a
source .env
set +a

exec caffeinate -d -i /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 main.py
