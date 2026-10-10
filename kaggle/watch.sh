#!/usr/bin/env bash
# usage: kaggle/watch.sh <name>   -- prints status on change; exits on terminal state
prev=""
while true; do
  s=$(python -m kaggle kernels status dev123123456/medleaf-esa-$1 2>&1 | tail -1)
  if [ "$s" != "$prev" ]; then echo "$(date +%H:%M) $s"; prev="$s"; fi
  echo "$s" | grep -qiE "complete|error|cancel|fail" && break
  sleep 60
done
