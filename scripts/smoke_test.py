#!/usr/bin/env python3
"""
Phase 1 smoke test: verify create/insert/search works on each database.
Run each DB independently so one failure doesn't block others.

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --db milvus       # test single db
  python scripts/smoke_test.py --db chroma,weaviate
"""
import argparse
import json
import time
import sys
import os
import numpy as np

VECTOR_DIM = 128
NUM_VECTORS = 100
NUM_QUERIES = 5
TOP_K = 3
SEED = 42

np.random.seed(SEED)
vectors = np.random.rand(NUM_VECTORS, VECTOR_DIM).astype(np.float32)
queries = np.random.rand(NUM_QUERIES, VECTOR_DIM).astype(np.float32)

results = []


def test_milvus():
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

    name = "smoke_test_collection"
    connections.connect(host="localhost", port=19530)

    # cleanup if exists
    if utility.has_collection(name):
        Collection(name).drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    schema = CollectionSchema(fields)
    col = Collection(name, schema)

    # insert
    ids = list(range(NUM_VECTORS))
    col.insert([ids, vectors.tolist()])
    col.flush()

    # index + load
    col.create_index("vector", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
    col.load()

    # search
    res = col.search(
        data=queries.tolist(),
        anns_field="vector",
        param={"metric_type": "L2"},
        limit=TOP_K,
    )

    result_count = sum(len(hits) for hits in res)

    # cleanup
    col.drop()
    connections.disconnect("default")

    return {"insert_count": NUM_VECTORS, "search_results": result_count}


def test_chroma():
    import chromadb

    client = chromadb.HttpClient(host="localhost", port=8000)
    name = "smoke_test_collection"

    # cleanup
    try:
        client.delete_collection(name)
    except Exception:
        pass

    col = client.create_collection(name=name, metadata={"hnsw:space": "l2"})

    # insert
    ids = [str(i) for i in range(NUM_VECTORS)]
    col.add(embeddings=vectors.tolist(), ids=ids)

    # search
    res = col.query(query_embeddings=queries.tolist(), n_results=TOP_K)
    result_count = sum(len(r) for r in res["ids"])

    # cleanup
    client.delete_collection(name)

    return {"insert_count": NUM_VECTORS, "search_results": result_count}


def test_weaviate():
    import weaviate

    client = weaviate.Client("http://localhost:8080")
    class_name = "SmokeTest"

    # cleanup
    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)

    # create class
    client.schema.create_class({
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "idx", "dataType": ["int"]}
        ],
    })

    # insert
    with client.batch as batch:
        for i, vec in enumerate(vectors):
            batch.add_data_object(
                data_object={"idx": i},
                class_name=class_name,
                vector=vec.tolist(),
            )

    # search
    result_count = 0
    for q in queries:
        res = (
            client.query.get(class_name, ["idx"])
            .with_near_vector({"vector": q.tolist()})
            .with_limit(TOP_K)
            .do()
        )
        hits = res.get("data", {}).get("Get", {}).get(class_name, [])
        result_count += len(hits)

    # cleanup
    client.schema.delete_class(class_name)

    return {"insert_count": NUM_VECTORS, "search_results": result_count}


DB_TESTS = {
    "milvus": test_milvus,
    "chroma": test_chroma,
    "weaviate": test_weaviate,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="milvus,chroma,weaviate",
                        help="Comma-separated list of databases to test")
    args = parser.parse_args()
    dbs = [d.strip() for d in args.db.split(",")]

    print(f"Smoke test: {dbs}\n")

    for db in dbs:
        fn = DB_TESTS.get(db)
        if not fn:
            print(f"[{db}] SKIP - unknown database")
            continue

        entry = {"db": db, "ok": False, "error": None}
        t0 = time.time()
        try:
            info = fn()
            entry.update(info)
            entry["ok"] = True
        except Exception as e:
            entry["error"] = str(e)
        entry["elapsed_sec"] = round(time.time() - t0, 3)

        status = "PASS" if entry["ok"] else "FAIL"
        print(f"[{db}] {status}  ({entry['elapsed_sec']}s)")
        if entry["error"]:
            print(f"  Error: {entry['error']}")
        results.append(entry)

    print(f"\n--- Summary ---")
    print(json.dumps(results, indent=2))

    if any(not r["ok"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
