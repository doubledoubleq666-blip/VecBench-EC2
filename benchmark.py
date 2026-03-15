import os
import time
import json
import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime

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
        """初始化数据库连接/集合"""
        raise NotImplementedError

    def insert(self, vectors: List[List[float]], ids: List[str] = None) -> float:
        """插入向量，返回耗时(秒)"""
        raise NotImplementedError

    def query(self, query_vectors: List[List[float]], top_k: int = 10) -> tuple[float, List[Any]]:
        """查询向量，返回(耗时, 结果)"""
        raise NotImplementedError

    def clear(self) -> None:
        """清理数据（用于重复实验）"""
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
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

        connections.connect(host=self.host, port=self.port)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "benchmark collection")
        self.collection = Collection(self.collection_name, schema, consistency_level="Strong")

    def insert(self, vectors, ids=None):
        if ids is None:
            ids = [i for i in range(len(vectors))]
        start = time.time()
        self.collection.insert([ids, vectors])
        self.collection.flush()
        # 创建索引（必须）
        index_params = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
        self.collection.create_index("vector", index_params)
        self.collection.load()
        return time.time() - start

    def query(self, query_vectors, top_k=10):
        start = time.time()
        res = self.collection.search(
            data=query_vectors,
            anns_field="vector",
            param={"metric_type": "L2"},
            limit=top_k
        )
        return time.time() - start, res

    def clear(self):
        if self.collection:
            self.collection.drop()

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
        self.collection = self.client.get_or_create_collection(name="benchmark_sift")

    def insert(self, vectors, ids=None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        start = time.time()
        self.collection.add(embeddings=vectors, ids=ids)
        return time.time() - start

    def query(self, query_vectors, top_k=10):
        start = time.time()
        res = self.collection.query(query_embeddings=query_vectors, n_results=top_k)
        return time.time() - start, res

    def clear(self):
        self.client.delete_collection(name="benchmark_sift")

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
        # 创建class
        class_obj = {
            "class": "BenchmarkSIFT",
            "vectorizer": "none",
            "properties": [{"name": "vector", "dataType": ["vector"]}]
        }
        if self.client.schema.exists("BenchmarkSIFT"):
            self.client.schema.delete_class("BenchmarkSIFT")
        self.client.schema.create_class(class_obj)

    def insert(self, vectors, ids=None):
        start = time.time()
        with self.client.batch as batch:
            for i, vec in enumerate(vectors):
                batch.add_data_object(
                    data_object={},
                    class_name="BenchmarkSIFT",
                    vector=vec
                )
        return time.time() - start

    def query(self, query_vectors, top_k=10):
        start = time.time()
        results = []
        for vec in query_vectors:
            res = self.client.query.get("BenchmarkSIFT").with_near_vector({
                "vector": vec
            }).with_limit(top_k).do()
            results.append(res)
        return time.time() - start, results

    def clear(self):
        if self.client.schema.exists("BenchmarkSIFT"):
            self.client.schema.delete_class("BenchmarkSIFT")

# ======================
# 基准测试执行器（核心）
# ======================
class BenchmarkRunner:
    def __init__(self, db: BaseVectorDB, db_type: str):
        self.db = db
        self.db_type = db_type
        self.results = []

    def run_benchmark(self,
                      vectors: List[List[float]],
                      query_vectors: List[List[float]],
                      batch_size: int = 1000,
                      repeat: int = 3,
                      top_k: int = 10,
                      dataset_size: str = "10K"):
        """
        运行完整基准测试：初始化 → 插入 → 查询 → 清理
        """
        print(f"\n===== 开始测试 {self.db_type} | 规模: {dataset_size} =====")

        for round_idx in range(repeat):
            print(f"\n第 {round_idx+1}/{repeat} 轮实验")

            # 1. 初始化
            self.db.init()
            time.sleep(1)

            # 2. 插入测试
            insert_time = self.db.insert(vectors)
            insert_qps = len(vectors) / insert_time if insert_time > 0 else 0
            print(f"插入耗时: {insert_time:.2f}s | QPS: {insert_qps:.2f}")

            # 3. 查询测试
            query_time, _ = self.db.query(query_vectors, top_k)
            query_latency_avg = query_time / len(query_vectors) if len(query_vectors) > 0 else 0
            query_qps = len(query_vectors) / query_time if query_time > 0 else 0
            print(f"查询总耗时: {query_time:.2f}s | 单查询延迟: {query_latency_avg:.4f}s | QPS: {query_qps:.2f}")

            # 4. 记录结果
            self.results.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "db": self.db_type,
                "dataset_size": dataset_size,
                "round": round_idx + 1,
                "vector_count": len(vectors),
                "query_count": len(query_vectors),
                "insert_time_sec": round(insert_time, 4),
                "insert_qps": round(insert_qps, 2),
                "query_total_time_sec": round(query_time, 4),
                "query_avg_latency_sec": round(query_latency_avg, 6),
                "query_qps": round(query_qps, 2),
                "top_k": top_k
            })

            # 5. 清理环境
            self.db.clear()
            time.sleep(1)

        print(f"\n✅ {self.db_type} 基准测试完成！")

    def save_results(self, output_dir: str = "results"):
        """保存为CSV"""
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.results)
        filename = f"{output_dir}/{self.db_type}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"\n📊 结果已保存: {filename}")
        return filename

# ======================
# 测试入口（可直接运行）
# ======================
if __name__ == "__main__":
    # ======================
    # 配置参数（你们修改这里）
    # ======================
    VECTOR_DIM = 128  # SIFT1M 固定维度
    TOP_K = 10
    REPEAT_TIMES = 3  # 每组实验重复3次
    DATASET_SCALE = "10K"  # 10K / 100K / 1M

    # 生成测试向量（正式使用时替换为真实SIFT1M数据）
    np.random.seed(42)  # 固定随机种子，保证公平
    test_vector_count = 10000  # 10K
    test_query_count = 100
    test_vectors = np.random.rand(test_vector_count, VECTOR_DIM).tolist()
    test_queries = np.random.rand(test_query_count, VECTOR_DIM).tolist()

    # ============= Milvus 测试 =============
    milvus = MilvusDB(host="localhost", port=19530, dim=VECTOR_DIM)
    runner_milvus = BenchmarkRunner(milvus, "milvus")
    runner_milvus.run_benchmark(
        vectors=test_vectors,
        query_vectors=test_queries,
        repeat=REPEAT_TIMES,
        top_k=TOP_K,
        dataset_size=DATASET_SCALE
    )
    runner_milvus.save_results()

    # ============= Chroma 测试 =============
    chroma = ChromaDB(host="localhost", port=8000, dim=VECTOR_DIM)
    runner_chroma = BenchmarkRunner(chroma, "chroma")
    runner_chroma.run_benchmark(
        vectors=test_vectors,
        query_vectors=test_queries,
        repeat=REPEAT_TIMES,
        top_k=TOP_K,
        dataset_size=DATASET_SCALE
    )
    runner_chroma.save_results()

    # ============= Weaviate 测试 =============
    weaviate = WeaviateDB(host="http://localhost:8080", dim=VECTOR_DIM)
    runner_weaviate = BenchmarkRunner(weaviate, "weaviate")
    runner_weaviate.run_benchmark(
        vectors=test_vectors,
        query_vectors=test_queries,
        repeat=REPEAT_TIMES,
        top_k=TOP_K,
        dataset_size=DATASET_SCALE
    )
    runner_weaviate.save_results()

    print("\n🎉 全部数据库基准测试完成！")