import os
import time
import uuid
import numpy as np
import pandas as pd
import psutil
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================
# 统一向量数据库基类
# ======================
class BaseVectorDB:
    def __init__(self, db_name: str, dim: int = 128):
        self.db_name = db_name
        self.dim = dim
        self.client = None
        self.collection = None
        self.results = []

    def init(self) -> None:
        raise NotImplementedError

    def insert(self, vectors: List[List[float]], ids: List[str] = None) -> float:
        raise NotImplementedError

    def query_single(self, query_vector: List[float], top_k: int = 10):
        """查询单个向量，返回结果。子类必须实现。"""
        raise NotImplementedError

    def query(self, query_vectors: List[List[float]], top_k: int = 10) -> tuple:
        """批量查询，返回 (总耗时, 每条延迟列表, 结果)"""
        latencies = []
        results = []
        for vec in query_vectors:
            t0 = time.time()
            res = self.query_single(vec, top_k)
            latencies.append(time.time() - t0)
            results.append(res)
        total = sum(latencies)
        return total, latencies, results

    def clear(self) -> None:
        pass

# ======================
# Milvus 适配器
# ======================
class MilvusDB(BaseVectorDB):
    def __init__(self, host="localhost", port=19530, dim=128):
        super().__init__("milvus", dim)
        self.host = host
        self.port = port
        self.collection_name = f"benchmark_sift_{dim}"

    def init(self):
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        connections.connect(host=self.host, port=self.port)
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "benchmark collection")
        self.collection = Collection(self.collection_name, schema, consistency_level="Strong")

    def insert(self, vectors, ids=None):
        if ids is None:
            ids = list(range(len(vectors)))
        start = time.time()
        batch_size = 10000
        for i in range(0, len(vectors), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vecs = vectors[i:i + batch_size]
            self.collection.insert([batch_ids, batch_vecs])
        self.collection.flush()
        index_params = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
        self.collection.create_index("vector", index_params)
        self.collection.load()
        return time.time() - start

    def query_single(self, query_vector, top_k=10):
        res = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2"},
            limit=top_k
        )
        return res

    def clear(self):
        from pymilvus import connections
        if self.collection:
            self.collection.drop()
        try:
            connections.disconnect("default")
        except Exception:
            pass

# ======================
# Chroma 适配器
# ======================
class ChromaDB(BaseVectorDB):
    def __init__(self, host="localhost", port=8000, dim=128):
        super().__init__("chroma", dim)
        self.host = host
        self.port = port

    def init(self):
        import chromadb
        self.client = chromadb.HttpClient(host=self.host, port=self.port)
        try:
            self.client.delete_collection(name="benchmark_sift")
        except Exception:
            pass
        self.collection = self.client.create_collection(name="benchmark_sift")

    def insert(self, vectors, ids=None):
        if ids is None:
            ids = [str(i) for i in range(len(vectors))]
        start = time.time()
        batch_size = 5000
        for i in range(0, len(vectors), batch_size):
            embeddings_batch = vectors[i:i + batch_size]
            ids_batch = ids[i:i + batch_size]
            self.collection.add(embeddings=embeddings_batch, ids=ids_batch)
        return time.time() - start

    def query_single(self, query_vector, top_k=10):
        res = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        return res

    def clear(self):
        try:
            self.client.delete_collection(name="benchmark_sift")
        except Exception:
            pass

# ======================
# Weaviate 适配器
# ======================
class WeaviateDB(BaseVectorDB):
    def __init__(self, host="http://localhost:8080", dim=128):
        super().__init__("weaviate", dim)
        self.host = host

    def init(self):
        import weaviate
        self.client = weaviate.Client(self.host)
        class_obj = {
            "class": "BenchmarkSIFT",
            "vectorizer": "none",
            "properties": [{"name": "payload", "dataType": ["text"]}]
        }
        if self.client.schema.exists("BenchmarkSIFT"):
            self.client.schema.delete_class("BenchmarkSIFT")
        self.client.schema.create_class(class_obj)

    def insert(self, vectors, ids=None):
        start = time.time()
        with self.client.batch as batch:
            for i, vec in enumerate(vectors):
                batch.add_data_object(
                    data_object={"payload": ""},
                    class_name="BenchmarkSIFT",
                    vector=vec
                )
        return time.time() - start

    def query_single(self, query_vector, top_k=10):
        res = self.client.query.get("BenchmarkSIFT", ["payload"]).with_near_vector({
            "vector": query_vector
        }).with_limit(top_k).do()
        return res

    def clear(self):
        if self.client.schema.exists("BenchmarkSIFT"):
            self.client.schema.delete_class("BenchmarkSIFT")


# ======================
# 并发查询执行器
# ======================
def run_concurrent_queries(db: BaseVectorDB, query_vectors, top_k, n_threads):
    """用多线程并发查询，返回每条查询的延迟列表"""
    latencies = []

    def _do_query(vec):
        t0 = time.time()
        db.query_single(vec, top_k)
        return time.time() - t0

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_do_query, vec) for vec in query_vectors]
        for f in as_completed(futures):
            latencies.append(f.result())

    return latencies


def compute_percentiles(latencies):
    """计算 P50/P95/P99"""
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "avg": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def get_system_snapshot():
    """获取当前系统资源快照"""
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": cpu,
        "mem_used_mb": round(mem.used / 1024 / 1024, 1),
        "mem_percent": mem.percent,
    }


# ======================
# 基准测试执行器（增强版）
# ======================
class BenchmarkRunner:
    def __init__(self, db: BaseVectorDB, db_type: str):
        self.db = db
        self.db_type = db_type
        self.results = []

    def run_benchmark(self,
                      vectors: List[List[float]],
                      query_vectors: List[List[float]],
                      repeat: int = 3,
                      top_k: int = 10,
                      dataset_size: str = "10K",
                      concurrency_levels: List[int] = None):
        if concurrency_levels is None:
            concurrency_levels = [1]

        print(f"\n===== Benchmark {self.db_type} | scale: {dataset_size} | concurrency: {concurrency_levels} =====")

        for round_idx in range(repeat):
            print(f"\n--- Round {round_idx+1}/{repeat} ---")

            # 1. Init
            self.db.init()
            time.sleep(1)

            # 2. System snapshot before insert
            sys_before = get_system_snapshot()

            # 3. Insert
            insert_time = self.db.insert(vectors)
            insert_qps = len(vectors) / insert_time if insert_time > 0 else 0
            print(f"  Insert: {insert_time:.2f}s | {insert_qps:.0f} vec/s")

            # 4. System snapshot after insert
            sys_after_insert = get_system_snapshot()

            # 5. Query at each concurrency level
            for n_threads in concurrency_levels:
                print(f"  Query (threads={n_threads}): ", end="", flush=True)

                if n_threads == 1:
                    # Sequential
                    _, latencies, _ = self.db.query(query_vectors, top_k)
                else:
                    latencies = run_concurrent_queries(self.db, query_vectors, top_k, n_threads)

                stats = compute_percentiles(latencies)
                total_time = sum(latencies)
                qps = len(query_vectors) / (max(latencies) * 1 if n_threads == 1 else (max(latencies))) if latencies else 0
                # For concurrent: throughput = total_queries / wall_clock_time
                wall_clock = max(latencies) if n_threads > 1 else total_time
                throughput = len(query_vectors) / wall_clock if wall_clock > 0 else 0

                print(f"avg={stats['avg']*1000:.1f}ms p50={stats['p50']*1000:.1f}ms "
                      f"p95={stats['p95']*1000:.1f}ms p99={stats['p99']*1000:.1f}ms "
                      f"QPS={throughput:.0f}")

                sys_after_query = get_system_snapshot()

                self.results.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "db": self.db_type,
                    "dataset_size": dataset_size,
                    "round": round_idx + 1,
                    "vector_count": len(vectors),
                    "query_count": len(query_vectors),
                    "concurrency": n_threads,
                    "top_k": top_k,
                    "insert_time_sec": round(insert_time, 4),
                    "insert_qps": round(insert_qps, 2),
                    "query_avg_ms": round(stats["avg"] * 1000, 3),
                    "query_p50_ms": round(stats["p50"] * 1000, 3),
                    "query_p95_ms": round(stats["p95"] * 1000, 3),
                    "query_p99_ms": round(stats["p99"] * 1000, 3),
                    "query_min_ms": round(stats["min"] * 1000, 3),
                    "query_max_ms": round(stats["max"] * 1000, 3),
                    "query_throughput_qps": round(throughput, 2),
                    "cpu_before": sys_before["cpu_percent"],
                    "cpu_after_insert": sys_after_insert["cpu_percent"],
                    "cpu_after_query": sys_after_query["cpu_percent"],
                    "mem_before_mb": sys_before["mem_used_mb"],
                    "mem_after_insert_mb": sys_after_insert["mem_used_mb"],
                    "mem_after_query_mb": sys_after_query["mem_used_mb"],
                })

            # 6. Cleanup
            self.db.clear()
            time.sleep(1)

        print(f"\n  Done: {self.db_type}")

    def save_results(self, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.results)
        filename = f"{output_dir}/{self.db_type}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"  Saved: {filename}")
        return filename


# ======================
# 入口
# ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector DB Benchmark")
    parser.add_argument("--db", default="milvus,chroma,weaviate",
                        help="Comma-separated list of databases")
    parser.add_argument("--dataset", default="data/processed/sift_10k.npy",
                        help="Path to vectors .npy file")
    parser.add_argument("--queries", default="data/processed/queries.npy",
                        help="Path to query vectors .npy file")
    parser.add_argument("--scale", default="10K",
                        help="Dataset scale label (10K/100K/1M)")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", default="1,4,8,16",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--num-queries", type=int, default=100,
                        help="Number of queries to use for latency measurement")
    parser.add_argument("--output", default="results",
                        help="Output directory for CSV results")
    args = parser.parse_args()

    VECTOR_DIM = 128
    concurrency_levels = [int(t) for t in args.threads.split(",")]

    print(f"Loading vectors from {args.dataset} ...")
    test_vectors = np.load(args.dataset).tolist()
    print(f"  Loaded {len(test_vectors)} vectors")

    print(f"Loading queries from {args.queries} ...")
    all_queries = np.load(args.queries).tolist()
    test_queries = all_queries[:args.num_queries]
    print(f"  Using {len(test_queries)} queries (from {len(all_queries)} available)")
    print(f"Concurrency levels: {concurrency_levels}")

    dbs_to_test = [d.strip() for d in args.db.split(",")]

    DB_CONFIGS = {
        "milvus": lambda: (MilvusDB(host="localhost", port=19530, dim=VECTOR_DIM), "milvus"),
        "chroma": lambda: (ChromaDB(host="localhost", port=8000, dim=VECTOR_DIM), "chroma"),
        "weaviate": lambda: (WeaviateDB(host="http://localhost:8080", dim=VECTOR_DIM), "weaviate"),
    }

    for db_name in dbs_to_test:
        if db_name not in DB_CONFIGS:
            print(f"Unknown database: {db_name}, skipping")
            continue
        db_instance, label = DB_CONFIGS[db_name]()
        runner = BenchmarkRunner(db_instance, label)
        runner.run_benchmark(
            vectors=test_vectors,
            query_vectors=test_queries,
            repeat=args.repeat,
            top_k=args.topk,
            dataset_size=args.scale,
            concurrency_levels=concurrency_levels,
        )
        runner.save_results(args.output)

    print("\nAll benchmarks completed!")
