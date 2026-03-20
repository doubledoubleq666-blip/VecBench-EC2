#!/usr/bin/env python3
"""
Read SIFT1M .fvecs/.ivecs files and generate 10K / 100K / 1M subsets as .npy files.
Also generates a fixed query set for fair comparison.

SIFT1M fvecs format:
  Each vector: 4 bytes (int32, dimension d) + d * 4 bytes (float32 values)
  For SIFT1M: d=128, so each record = 4 + 128*4 = 516 bytes
"""
import os
import sys
import json
import struct
import numpy as np
from datetime import datetime

SEED = 42
SIFT_DIM = 128
SUBSETS = {
    "sift_10k": 10_000,
    "sift_100k": 100_000,
    "sift_1m": 1_000_000,
}
QUERY_COUNT = 1000

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
RAW_DIR = os.path.join(PROJECT_DIR, "sift1m_data")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")


def read_fvecs(filepath: str, max_count: int = None) -> np.ndarray:
    """Read .fvecs file and return numpy array of shape (n, dim)."""
    vectors = []
    with open(filepath, "rb") as f:
        count = 0
        while True:
            if max_count and count >= max_count:
                break
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            if len(vec) != dim:
                break
            vectors.append(vec)
            count += 1
    return np.array(vectors, dtype=np.float32)


def read_ivecs(filepath: str, max_count: int = None) -> np.ndarray:
    """Read .ivecs file and return numpy array of shape (n, k)."""
    vectors = []
    with open(filepath, "rb") as f:
        count = 0
        while True:
            if max_count and count >= max_count:
                break
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
            if len(vec) != dim:
                break
            vectors.append(vec)
            count += 1
    return np.array(vectors, dtype=np.int32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base_path = os.path.join(RAW_DIR, "sift_base.fvecs")
    query_path = os.path.join(RAW_DIR, "sift_query.fvecs")
    gt_path = os.path.join(RAW_DIR, "sift_groundtruth.ivecs")

    if not os.path.exists(base_path):
        print(f"ERROR: {base_path} not found. Run downloadDataset.py first.")
        sys.exit(1)

    # Read base vectors (full 1M)
    print(f"Reading base vectors from {base_path} ...")
    base_vectors = read_fvecs(base_path)
    print(f"  Loaded {base_vectors.shape[0]} vectors, dim={base_vectors.shape[1]}")

    # Read query vectors
    print(f"Reading query vectors from {query_path} ...")
    query_vectors = read_fvecs(query_path)
    print(f"  Loaded {query_vectors.shape[0]} query vectors")

    # Read ground truth
    if os.path.exists(gt_path):
        print(f"Reading ground truth from {gt_path} ...")
        groundtruth = read_ivecs(gt_path)
        np.save(os.path.join(OUT_DIR, "groundtruth.npy"), groundtruth)
        print(f"  Saved groundtruth.npy ({groundtruth.shape})")

    # Save query set (use first QUERY_COUNT queries, fixed)
    np.random.seed(SEED)
    q = query_vectors[:QUERY_COUNT]
    np.save(os.path.join(OUT_DIR, "queries.npy"), q)
    print(f"  Saved queries.npy ({q.shape})")

    # Generate subsets
    for name, count in SUBSETS.items():
        if count > base_vectors.shape[0]:
            print(f"  WARN: requested {count} but only {base_vectors.shape[0]} available, using all")
            subset = base_vectors
        else:
            subset = base_vectors[:count]
        out_path = os.path.join(OUT_DIR, f"{name}.npy")
        np.save(out_path, subset)
        print(f"  Saved {name}.npy ({subset.shape})")

    # Metadata
    meta = {
        "seed": SEED,
        "dim": SIFT_DIM,
        "query_count": QUERY_COUNT,
        "subsets": {k: v for k, v in SUBSETS.items()},
        "base_total": int(base_vectors.shape[0]),
        "created": datetime.now().isoformat(),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. All files saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
