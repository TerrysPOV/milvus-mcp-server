from server import create_collection, insert_data, create_index, load_collection, search

# 1. Create a collection
print(create_collection("test_collection"))

# 2. Insert some dummy data (must match the expected dimension, e.g. 1536)
dummy_vectors = [[0.01 * i for i in range(1536)] for _ in range(10)]
print(insert_data("test_collection", dummy_vectors))

# 3. Create index
print(create_index("test_collection"))

# 4. Load collection
print(load_collection("test_collection"))

# 5. Search with a sample query
query = [0.01 * i for i in range(1536)]
results = search("test_collection", query)
print("Search results:")
print(results)
