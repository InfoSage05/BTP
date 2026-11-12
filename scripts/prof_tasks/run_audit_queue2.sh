#!/bin/bash
cd /Users/mauryavardhansingh/Desktop/BTP/BTP/scripts/prof_tasks
R=/Users/mauryavardhansingh/Desktop/BTP/BTP/results/prof_tasks
while pgrep -f "run_audit_queue.sh" > /dev/null; do sleep 20; done
echo "### fluid transfer $(date)"
python3 -u audit_fluid_transfer.py > $R/audit_fluid_transfer.log 2>&1
echo "### fluid transfer done rc=$?"
echo "=== ALL AUDITS COMPLETE $(date) ==="
