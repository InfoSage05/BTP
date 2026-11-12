#!/bin/bash
# Run the remaining audits sequentially, after the feature audit finishes.
cd /Users/mauryavardhansingh/Desktop/BTP/BTP/scripts/prof_tasks
R=/Users/mauryavardhansingh/Desktop/BTP/BTP/results/prof_tasks
while pgrep -f "audit_v2.py" > /dev/null; do sleep 20; done
echo "=== QUEUE START $(date) ==="
echo "### 1/3 OOD conformal"
python3 -u audit_ood_conformal.py  > $R/audit_ood_conformal.log 2>&1
echo "### 1/3 done rc=$?"
echo "### 2/3 Katto C sensitivity"
python3 -u audit_katto_c.py        > $R/audit_katto_c.log 2>&1
echo "### 2/3 done rc=$?"
echo "### 3/3 definitive re-run with fixed LUT"
python3 -u final_review.py         > $R/final_review_v2.log 2>&1
echo "### 3/3 done rc=$?"
echo "=== QUEUE COMPLETE $(date) ==="
