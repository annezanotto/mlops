#!/bin/bash

# Get absolute path to the Python interpreter in your current venv
PYTHON_PATH="$(which python3)"
# Get absolute path to the script
SCRIPT_PATH="/Users/andre/Projects/mlflow/monitor_retrainer.py"
# Log file path
LOG_PATH="/Users/andre/Projects/mlflow/monitor_retrainer.log"

# Write out current crontab and add new job
(crontab -l 2>/dev/null; echo "0 2 * * * cd /Users/andre/Projects/mlflow && $PYTHON_PATH $SCRIPT_PATH >> $LOG_PATH 2>&1") | crontab -

echo "✅ Cron job added: monitor_retrainer.py will run every day at 2:00 AM."