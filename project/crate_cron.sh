#!/bin/bash

# Get absolute path to the Python interpreter in your current venv
PYTHON_PATH="$(which python3)"

# Get absolute path to the script (in the same directory as this .sh)
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/monitor_retrainer.py"

# Log file path (in the same directory as this .sh)
LOG_PATH="$(cd "$(dirname "$0")" && pwd)/monitor_retrainer.log"

# Write out current crontab and add new job
(crontab -l 2>/dev/null; echo "0 2 * * * $PYTHON_PATH $SCRIPT_PATH >> $LOG_PATH 2>&1") | crontab -

echo "✅ Cron job added: monitor_retrainer.py will run every day at 2:00 AM."