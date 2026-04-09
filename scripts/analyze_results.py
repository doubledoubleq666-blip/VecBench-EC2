#!/usr/bin/env python3
"""
Phase 4: Analyze benchmark results and generate visualizations.
Reads all Phase 3 CSV files, aggregates across rounds, produces charts and TCO analysis.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ============================================================
# 1. Load and merge all Phase 3 CSVs
# ============================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'bench', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'bench', 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Phase 3 files (timestamped 20260411, with concurrency data = file size > 1KB)
phase3_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_benchmark_20260411_*.csv')))
phase3_files = [f for f in phase3_files if os.path.getsize(f) > 1000]  # skip Phase 2 leftovers

print(f"Loading {len(phase3_files)} Phase 3 CSV files...")
dfs = []
for f in phase3_files:
    df = pd.read_csv(f)
    dfs.append(df)
    print(f"  {os.path.basename(f)}: {len(df)} rows, db={df['db'].iloc[0]}, scale={df['dataset_size'].iloc[0]}")

data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(data)}")
print(f"DBs: {data['db'].unique()}")
print(f"Scales: {data['dataset_size'].unique()}")
print(f"Concurrency levels: {sorted(data['concurrency'].unique())}")

# ============================================================
# 2. Aggregate: mean across 3 rounds
# ============================================================
agg_cols = [
    'insert_time_sec', 'insert_qps',
    'query_avg_ms', 'query_p50_ms', 'query_p95_ms', 'query_p99_ms',
    'query_min_ms', 'query_max_ms', 'query_throughput_qps',
    'cpu_after_insert', 'cpu_after_query',
    'mem_after_insert_mb', 'mem_after_query_mb',
]

agg = data.groupby(['db', 'dataset_size', 'concurrency'])[agg_cols].mean().reset_index()

# Sort dataset_size properly
size_order = {'10K': 0, '100K': 1, '1M': 2}
agg['_size_order'] = agg['dataset_size'].map(size_order)
agg = agg.sort_values(['db', '_size_order', 'concurrency']).reset_index(drop=True)

print("\n=== Aggregated Results (mean of 3 rounds) ===")
print(agg.to_string(index=False))

# Save aggregated CSV
agg_csv = os.path.join(OUTPUT_DIR, 'aggregated_results.csv')
agg.drop(columns='_size_order').to_csv(agg_csv, index=False)
print(f"\nSaved: {agg_csv}")

# ============================================================
# 3. Color & style config
# ============================================================
DB_COLORS = {'milvus': '#1f77b4', 'chroma': '#2ca02c', 'weaviate': '#ff7f0e'}
DB_LABELS = {'milvus': 'Milvus', 'chroma': 'Chroma', 'weaviate': 'Weaviate'}
SCALE_LABELS = ['10K', '100K', '1M']
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {path}")


# ============================================================
# 4. Chart 1: Insert Throughput (vectors/s) by scale
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.25
scales_available = sorted(agg['dataset_size'].unique(), key=lambda x: size_order[x])

# For insert, take concurrency=1 rows (insert is always sequential)
insert_data = agg[agg['concurrency'] == 1]

for i, db in enumerate(['milvus', 'chroma', 'weaviate']):
    db_data = insert_data[insert_data['db'] == db]
    scales = db_data['dataset_size'].tolist()
    qps_vals = db_data['insert_qps'].tolist()
    x_pos = [list(scales_available).index(s) + i * bar_width for s in scales]
    bars = ax.bar(x_pos, qps_vals, bar_width, label=DB_LABELS[db], color=DB_COLORS[db])
    for bar, val in zip(bars, qps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Dataset Scale')
ax.set_ylabel('Insert Throughput (vectors/s)')
ax.set_title('Insert Throughput Comparison')
ax.set_xticks([j + bar_width for j in range(len(scales_available))])
ax.set_xticklabels(scales_available)
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_fig(fig, 'insert_throughput.png')

# ============================================================
# 5. Chart 2: Insert Time (seconds) by scale
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for i, db in enumerate(['milvus', 'chroma', 'weaviate']):
    db_data = insert_data[insert_data['db'] == db]
    scales = db_data['dataset_size'].tolist()
    time_vals = db_data['insert_time_sec'].tolist()
    x_pos = [list(scales_available).index(s) + i * bar_width for s in scales]
    bars = ax.bar(x_pos, time_vals, bar_width, label=DB_LABELS[db], color=DB_COLORS[db])
    for bar, val in zip(bars, time_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Dataset Scale')
ax.set_ylabel('Insert Time (seconds)')
ax.set_title('Insert Time Comparison')
ax.set_xticks([j + bar_width for j in range(len(scales_available))])
ax.set_xticklabels(scales_available)
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_fig(fig, 'insert_time.png')

# ============================================================
# 6. Chart 3: Query Latency (avg, P50, P95, P99) - single thread
# ============================================================
fig, axes = plt.subplots(1, len(scales_available), figsize=(5*len(scales_available), 5), sharey=False)
if len(scales_available) == 1:
    axes = [axes]

for idx, scale in enumerate(scales_available):
    ax = axes[idx]
    single = agg[(agg['concurrency'] == 1) & (agg['dataset_size'] == scale)]
    dbs_present = single['db'].tolist()

    metrics = ['query_avg_ms', 'query_p50_ms', 'query_p95_ms', 'query_p99_ms']
    metric_labels = ['Avg', 'P50', 'P95', 'P99']
    x = np.arange(len(metric_labels))
    w = 0.22

    for j, db in enumerate(dbs_present):
        row = single[single['db'] == db].iloc[0]
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + j * w, vals, w, label=DB_LABELS[db], color=DB_COLORS[db])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_title(f'Query Latency @ {scale} (1 thread)')
    ax.set_xticks(x + w)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Latency (ms)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Query Latency Distribution (Sequential)', fontsize=13, y=1.02)
save_fig(fig, 'query_latency_sequential.png')

# ============================================================
# 7. Chart 4: Concurrency Scaling - QPS vs threads
# ============================================================
fig, axes = plt.subplots(1, len(scales_available), figsize=(5*len(scales_available), 5), sharey=False)
if len(scales_available) == 1:
    axes = [axes]

threads = sorted(agg['concurrency'].unique())

for idx, scale in enumerate(scales_available):
    ax = axes[idx]
    scale_data = agg[agg['dataset_size'] == scale]
    for db in ['milvus', 'chroma', 'weaviate']:
        db_data = scale_data[scale_data['db'] == db]
        if len(db_data) == 0:
            continue
        ax.plot(db_data['concurrency'], db_data['query_throughput_qps'],
                marker='o', label=DB_LABELS[db], color=DB_COLORS[db], linewidth=2)

    ax.set_xlabel('Concurrency (threads)')
    ax.set_ylabel('Query Throughput (QPS)')
    ax.set_title(f'Concurrency Scaling @ {scale}')
    ax.set_xticks(threads)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle('Query Throughput vs Concurrency', fontsize=13, y=1.02)
save_fig(fig, 'concurrency_scaling.png')

# ============================================================
# 8. Chart 5: P95 Latency under concurrency
# ============================================================
fig, axes = plt.subplots(1, len(scales_available), figsize=(5*len(scales_available), 5), sharey=False)
if len(scales_available) == 1:
    axes = [axes]

for idx, scale in enumerate(scales_available):
    ax = axes[idx]
    scale_data = agg[agg['dataset_size'] == scale]
    for db in ['milvus', 'chroma', 'weaviate']:
        db_data = scale_data[scale_data['db'] == db]
        if len(db_data) == 0:
            continue
        ax.plot(db_data['concurrency'], db_data['query_p95_ms'],
                marker='s', label=DB_LABELS[db], color=DB_COLORS[db], linewidth=2)

    ax.set_xlabel('Concurrency (threads)')
    ax.set_ylabel('P95 Latency (ms)')
    ax.set_title(f'P95 Tail Latency @ {scale}')
    ax.set_xticks(threads)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle('P95 Latency vs Concurrency', fontsize=13, y=1.02)
save_fig(fig, 'p95_latency_concurrency.png')

# ============================================================
# 9. Chart 6: Memory Usage after insert
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
mem_data = insert_data.copy()

for i, db in enumerate(['milvus', 'chroma', 'weaviate']):
    db_data = mem_data[mem_data['db'] == db]
    scales = db_data['dataset_size'].tolist()
    mem_vals = db_data['mem_after_insert_mb'].tolist()
    x_pos = [list(scales_available).index(s) + i * bar_width for s in scales]
    bars = ax.bar(x_pos, mem_vals, bar_width, label=DB_LABELS[db], color=DB_COLORS[db])
    for bar, val in zip(bars, mem_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Dataset Scale')
ax.set_ylabel('Memory Used (MB)')
ax.set_title('Memory Usage After Ingestion')
ax.set_xticks([j + bar_width for j in range(len(scales_available))])
ax.set_xticklabels(scales_available)
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_fig(fig, 'memory_usage.png')

# ============================================================
# 10. Chart 7: Comprehensive comparison heatmap (normalized)
# ============================================================
# Compare at 100K scale, 1 thread (all 3 DBs available)
compare = agg[(agg['dataset_size'] == '100K') & (agg['concurrency'] == 1)].copy()
compare = compare.set_index('db')

metrics_for_heatmap = {
    'Insert Time (s)': 'insert_time_sec',
    'Insert QPS': 'insert_qps',
    'Query Avg (ms)': 'query_avg_ms',
    'Query P95 (ms)': 'query_p95_ms',
    'Query P99 (ms)': 'query_p99_ms',
    'Memory (MB)': 'mem_after_insert_mb',
}

heatmap_data = pd.DataFrame()
for label, col in metrics_for_heatmap.items():
    heatmap_data[label] = compare[col]

fig, ax = plt.subplots(figsize=(10, 4))
heatmap_arr = heatmap_data.values.astype(float)

# Normalize each column to [0, 1]
norm_data = (heatmap_arr - heatmap_arr.min(axis=0)) / (heatmap_arr.max(axis=0) - heatmap_arr.min(axis=0) + 1e-9)

im = ax.imshow(norm_data, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(metrics_for_heatmap)))
ax.set_xticklabels(list(metrics_for_heatmap.keys()), rotation=30, ha='right')
ax.set_yticks(range(len(compare)))
ax.set_yticklabels([DB_LABELS.get(db, db) for db in compare.index])

# Add value annotations
for i in range(len(compare)):
    for j in range(len(metrics_for_heatmap)):
        val = heatmap_arr[i, j]
        text = f'{val:,.1f}' if val < 10000 else f'{val:,.0f}'
        ax.text(j, i, text, ha='center', va='center', fontsize=9,
                color='white' if norm_data[i, j] > 0.6 else 'black')

ax.set_title('Performance Comparison at 100K Scale (1 thread) — Lower is Better (except Insert QPS)')
fig.colorbar(im, ax=ax, shrink=0.8)
save_fig(fig, 'comparison_heatmap.png')

# ============================================================
# 11. TCO Calculation
# ============================================================
print("\n" + "="*60)
print("AWS Total Cost of Ownership (TCO) Analysis")
print("="*60)

# t2.large pricing (us-east-1, on-demand)
EC2_HOURLY = 0.0928  # USD/hour
EC2_MONTHLY = EC2_HOURLY * 730  # ~$67.74
EBS_GB_MONTHLY = 0.08  # gp3 USD/GB/month
EBS_SIZE_GB = 30
EBS_MONTHLY = EBS_GB_MONTHLY * EBS_SIZE_GB  # $2.40

# Data transfer (minimal for benchmark, ~1GB outbound)
DATA_TRANSFER_GB = 1
DATA_TRANSFER_MONTHLY = 0.09 * DATA_TRANSFER_GB  # $0.09/GB

print(f"\nEC2 Instance: t2.large (2 vCPU, 8 GB RAM)")
print(f"  On-demand: ${EC2_HOURLY:.4f}/hr = ${EC2_MONTHLY:.2f}/month")
print(f"EBS Storage: {EBS_SIZE_GB} GB gp3")
print(f"  Cost: ${EBS_MONTHLY:.2f}/month")
print(f"Data Transfer: ~{DATA_TRANSFER_GB} GB outbound")
print(f"  Cost: ${DATA_TRANSFER_MONTHLY:.2f}/month")
print(f"\nTotal Monthly TCO: ${EC2_MONTHLY + EBS_MONTHLY + DATA_TRANSFER_MONTHLY:.2f}")

# Per-database operational cost (based on benchmark runtime)
print("\n--- Per-Database Benchmark Runtime ---")
for db in ['milvus', 'chroma', 'weaviate']:
    db_rows = data[data['db'] == db]
    if len(db_rows) == 0:
        continue
    # Total insert time across all scales and rounds
    insert_total = db_rows.groupby(['dataset_size', 'round'])['insert_time_sec'].first().sum()
    # Estimate total runtime including queries
    first_ts = pd.to_datetime(db_rows['timestamp'].min())
    last_ts = pd.to_datetime(db_rows['timestamp'].max())
    runtime_min = (last_ts - first_ts).total_seconds() / 60
    runtime_cost = (runtime_min / 60) * EC2_HOURLY
    print(f"  {DB_LABELS[db]:10s}: ~{runtime_min:.0f} min runtime, "
          f"insert total={insert_total:.1f}s, "
          f"EC2 cost=${runtime_cost:.3f}")

# Docker resource overhead comparison
print("\n--- Docker Resource Footprint ---")
print("  Milvus: 3 containers (etcd + minio + standalone), ~5-7 GB RAM at 1M")
print("  Chroma: 1 container, ~1.4-1.6 GB RAM at 100K, OOM at 1M")
print("  Weaviate: 1 container, ~1.4-1.5 GB RAM at 100K, OOM/timeout at 1M")

# TCO summary table
tco_data = {
    'Component': ['EC2 (t2.large)', 'EBS (30GB gp3)', 'Data Transfer', 'Total'],
    'Monthly Cost (USD)': [f'${EC2_MONTHLY:.2f}', f'${EBS_MONTHLY:.2f}',
                           f'${DATA_TRANSFER_MONTHLY:.2f}',
                           f'${EC2_MONTHLY + EBS_MONTHLY + DATA_TRANSFER_MONTHLY:.2f}'],
}
tco_df = pd.DataFrame(tco_data)
tco_csv = os.path.join(OUTPUT_DIR, 'tco_analysis.csv')
tco_df.to_csv(tco_csv, index=False)
print(f"\nSaved: {tco_csv}")

# ============================================================
# 12. Summary table
# ============================================================
print("\n" + "="*60)
print("SUMMARY: Key Findings")
print("="*60)

# Best at each scale for sequential queries
for scale in scales_available:
    s_data = agg[(agg['dataset_size'] == scale) & (agg['concurrency'] == 1)]
    if len(s_data) == 0:
        continue
    best_insert = s_data.loc[s_data['insert_qps'].idxmax()]
    best_query = s_data.loc[s_data['query_avg_ms'].idxmin()]
    print(f"\n  [{scale}]")
    print(f"    Best insert throughput: {DB_LABELS[best_insert['db']]} ({best_insert['insert_qps']:,.0f} vec/s)")
    print(f"    Best query latency:     {DB_LABELS[best_query['db']]} ({best_query['query_avg_ms']:.1f} ms avg)")

# Scalability
print("\n  [Scalability to 1M]")
print("    Milvus:   SUCCESS - 35,000+ vec/s insert, queries work at all concurrency levels")
print("    Chroma:   FAILED  - OOM during 1M insert (>5.3 GB memory)")
print("    Weaviate: FAILED  - Killed during 1M batch insert (ReadTimeout + OOM)")

print("\n  [Concurrency Winner at 100K]")
for threads in [1, 4, 8, 16]:
    t_data = agg[(agg['dataset_size'] == '100K') & (agg['concurrency'] == threads)]
    if len(t_data) == 0:
        continue
    best = t_data.loc[t_data['query_throughput_qps'].idxmax()]
    print(f"    {threads:2d} threads: {DB_LABELS[best['db']]} ({best['query_throughput_qps']:,.0f} QPS)")

print("\nPhase 4 analysis complete!")
