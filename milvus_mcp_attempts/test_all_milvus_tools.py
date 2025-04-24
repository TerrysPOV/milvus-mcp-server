# test_all_milvus_tools.py

from milvus_tools import (
    create_collection,
    insert_data,
    create_index,
    load_collection,
    search,
)
import random

collection_name = "demo_collection"

# Dummy data for testing
dim = 8
num_vectors = 10
ids = list(range(num_vectors))
embeddings = [[random.uniform(0, 1) for _ in range(dim)] for _ in range(num_vectors)]
query_vector = embeddings[0]  # Use the first as a test query

print("🧪 1. Creating collection...")
print(create_collection(collection_name, dim))

print("🧪 2. Inserting data...")
print(insert_data(collection_name, ids, embeddings))

print("🧪 3. Creating index...")
print(create_index(collection_name))

print("🧪 4. Loading collection...")
print(load_collection(collection_name))

print("🧪 5. Running search...")
results = search(collection_name, query_vector)
print("🔍 Search results:")
for hit_id, distance in results:
    print(f"ID: {hit_id}, Distance: {distance}")
