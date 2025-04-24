from milvus_tools import (
    create_collection,
    insert_data,
    create_index,
    load_collection,
    search
)

collection_name = "demo_collection"
dim = 8

print(create_collection(collection_name, dim))

# Sample data
ids = list(range(10))
embeddings = [[float(i + j) for j in range(dim)] for i in range(len(ids))]

print(insert_data(collection_name, ids, embeddings))
print(create_index(collection_name))
print(load_collection(collection_name))

print("🔍 Search results:")
query = [1.0] * dim
results = search(collection_name, query)
for id, distance in results:
    print(f"ID: {id}, Distance: {distance}")
