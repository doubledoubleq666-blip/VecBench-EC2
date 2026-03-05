# Performance Benchmarking of Open-Source Vector Databases for RAG Systems on AWS EC2

CS5296 Cloud Computing, Spring 2026 — Technical Project

## Team

| Name | Student ID |
|------|-----------|
| QING Qiguo | 59905956 |
| LUO Yuxi | 59570433 |

## Project Overview

This project benchmarks three open-source vector databases — **Milvus**, **Chroma**, and **Weaviate** — on AWS EC2, evaluating ingestion speed, query latency, scalability under concurrent load, resource consumption, and total cost of ownership (TCO).

**Dataset:** SIFT1M (1M vectors, 128 dimensions)
**Test scales:** 10K / 100K / 1M vectors

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── benchmark.py              # Benchmark framework (clients + runner)
├── downloadDataset.py        # Download SIFT1M from HuggingFace
├── docker/
│   ├── chroma/docker-compose.yml
│   ├── weaviate/docker-compose.yml
│   └── milvus/docker-compose.yml
├── scripts/
│   ├── setup_env.sh          # EC2 one-click environment setup
│   ├── prepare_sift_subsets.py   # Convert fvecs → npy subsets
│   ├── smoke_test.py         # Quick connectivity test for all DBs
│   └── collect_system_metrics.py # CPU/mem/disk monitoring
├── bench/results/            # CSV output from benchmark runs
├── data/
│   ├── raw/                  # Original SIFT1M fvecs
│   └── processed/            # 10K/100K/1M .npy subsets
└── docs/
```

## Quick Start

### 1. EC2 Environment Setup

```bash
# On a fresh Ubuntu 22.04 EC2 instance:
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
# Log out and back in for Docker group changes, then:
source venv/bin/activate
```

### 2. Download and Prepare Dataset

```bash
python downloadDataset.py
python scripts/prepare_sift_subsets.py
```

This generates:
- `data/processed/sift_10k.npy` (10K vectors)
- `data/processed/sift_100k.npy` (100K vectors)
- `data/processed/sift_1m.npy` (1M vectors)
- `data/processed/queries.npy` (1K query vectors)

### 3. Start a Database

Due to t2.large memory limits, run **one database at a time**:

```bash
# Chroma
cd docker/chroma && docker compose up -d && cd ../..

# Weaviate
cd docker/weaviate && docker compose up -d && cd ../..

# Milvus (heaviest — needs etcd + minio)
cd docker/milvus && docker compose up -d && cd ../..
```

### 4. Run Smoke Test

```bash
python scripts/smoke_test.py              # all three
python scripts/smoke_test.py --db chroma  # single
```

### 5. Run Benchmark

```bash
python benchmark.py
```

### 6. Collect System Metrics (in a separate terminal)

```bash
python scripts/collect_system_metrics.py --duration 300 --output bench/results/metrics.csv
```

## Fairness Rules

All benchmarks follow these rules to ensure fair comparison:

1. **Same EC2 instance** — identical hardware for every database
2. **Same OS & Docker version** — Ubuntu 22.04, Docker CE latest
3. **Same dataset** — SIFT1M subsets from identical .npy files
4. **Same query set** — fixed 1K queries from `queries.npy`
5. **Fixed random seed** — `seed=42` for all randomized operations
6. **Same top-k** — `k=10` for all search benchmarks
7. **Warm-up** — 1 second pause after init before measuring
8. **Repetition** — each experiment repeated 3 times, report mean
9. **Sequential execution** — one database at a time to avoid resource contention
10. **Fresh state** — collection dropped and recreated between rounds

## Known Constraints

- **AWS Academy Learner Lab denies c5.xlarge**. Phase 1 uses `t2.large` (2 vCPU, 8 GB RAM) as a smoke-test environment. Final benchmarks may use a larger allowed instance type.
- **t2.large has limited memory** — run one database at a time, especially Milvus.
- Phase 1 results are for validation only, not final performance conclusions.

## Metrics Collected

| Category | Metrics |
|----------|---------|
| Ingestion | Total time (s), throughput (vectors/s) |
| Query | Total time (s), avg latency (s), QPS |
| System | CPU %, memory %, disk read/write MB |
| Cost | EC2 compute, EBS storage, data transfer (monthly TCO) |

## Tech Stack

- **Vector DBs:** Milvus v2.4, Chroma latest, Weaviate v1.24
- **Infra:** AWS EC2, Docker, Docker Compose
- **Language:** Python 3.10+
- **Libraries:** pymilvus, chromadb, weaviate-client v3, numpy, pandas, psutil
