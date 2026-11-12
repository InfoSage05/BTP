#!/bin/bash
cd /Users/mauryavardhansingh/Desktop/BTP/BTP/scripts/prof_tasks
R=/Users/mauryavardhansingh/Desktop/BTP/BTP/results/prof_tasks
while pgrep -f "run_audit_queue2.sh" > /dev/null; do sleep 20; done
echo "### strict OOD conformal $(date)"
python3 -u audit_ood_conformal_strict.py > $R/audit_ood_strict.log 2>&1
echo "### strict done rc=$?"
echo "=== EVERYTHING COMPLETE $(date) ==="
