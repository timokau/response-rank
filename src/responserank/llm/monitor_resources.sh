#!/bin/bash
# Resource monitoring script for HPC jobs that prints to stdout

INTERVAL=60

echo "====== RESOURCE MONITORING STARTED ======"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Monitoring interval: ${INTERVAL}s"
echo "========================================"

monitor_count=0

while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    monitor_count=$((monitor_count + 1))

    echo ""
    echo "====== RESOURCE SNAPSHOT #${monitor_count} at ${timestamp} ======"

    # GPU monitoring
    if command -v nvidia-smi &> /dev/null; then
        echo "----- GPU MEMORY -----"
        nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv
    fi

    # CPU memory monitoring
    echo "----- CPU MEMORY -----"
    free -h

    # Top memory processes (5 most memory-intensive)
    echo "----- TOP MEMORY PROCESSES -----"
    ps -eo pid,%mem,%cpu,rss:10,cmd --sort=-%mem | head -6

    echo "====== END SNAPSHOT #${monitor_count} ======"
    echo ""

    sleep "$INTERVAL"
done
