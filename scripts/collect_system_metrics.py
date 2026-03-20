#!/usr/bin/env python3
"""
Collect system metrics (CPU, memory, disk) at 1-second intervals.
Writes to CSV for later analysis.

Usage:
  python scripts/collect_system_metrics.py --output bench/results/metrics.csv --duration 300
  python scripts/collect_system_metrics.py --duration 0   # run until Ctrl+C
"""
import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime

import psutil

running = True


def signal_handler(sig, frame):
    global running
    running = False
    print("\nStopping metrics collection...")


signal.signal(signal.SIGINT, signal_handler)


def collect_once() -> dict:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu,
        "mem_total_mb": round(mem.total / 1024 / 1024, 1),
        "mem_used_mb": round(mem.used / 1024 / 1024, 1),
        "mem_percent": mem.percent,
        "disk_read_mb": round(disk.read_bytes / 1024 / 1024, 1) if disk else 0,
        "disk_write_mb": round(disk.write_bytes / 1024 / 1024, 1) if disk else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench/results/system_metrics.csv")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duration in seconds (0 = until Ctrl+C)")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fields = ["timestamp", "cpu_percent", "mem_total_mb", "mem_used_mb",
              "mem_percent", "disk_read_mb", "disk_write_mb"]

    # Prime the CPU counter
    psutil.cpu_percent(interval=None)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        start = time.time()
        count = 0
        print(f"Collecting metrics to {args.output} (interval={args.interval}s) ...")

        while running:
            if args.duration > 0 and (time.time() - start) >= args.duration:
                break
            row = collect_once()
            writer.writerow(row)
            f.flush()
            count += 1
            time.sleep(args.interval)

    print(f"Done. Collected {count} samples to {args.output}")


if __name__ == "__main__":
    main()
