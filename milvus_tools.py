import time
import logging
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus_tools")

# Connect to Milvus
for i in range(10):
    try:
        connections.connect(alias="default", host="standalone", port="19530")
        logger.info("Connected to Milvus at standalone:19530")
        break
    except Exception as e:
        logger.warning(f"Waiting for Milvus... ({i+1}/10): {e}")
        time.sleep(2)
else:
    raise ConnectionError("Failed to connect to Milvus after multiple attempts.")

# Schema builder
def get_default_schema(dim: int = 1536):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    return CollectionSchema(fields, description="Default schema")

# Create collection
def create_collection(name: str, dim: int = 1536) -> str:
    if name in utility.list_collections():
        return f"Collection '{name}' already exists."
    schema = get_default_schema(dim)
    Collection(name, schema)
    return f"Collection '{name}' created."

# Insert data
def insert_data(collection_name: str, vectors: list) -> str:
    collection = Collection(collection_name)
    entities = [vectors]
    collection.insert(entities)
    return f"Inserted {len(vectors)} vectors into '{collection_name}'."

# Create index
def create_index(collection_name: str) -> str:
    collection = Collection(collection_name)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return f"Index created on '{collection_name}'."

# Load collection
def load_collection(collection_name: str) -> str:
    collection = Collection(collection_name)
    collection.load()
    return f"Collection '{collection_name}' loaded."

# Search vectors
def search(collection_name: str, query_vector: list, top_k: int = 5) -> list:
    collection = Collection(collection_name)
    if not collection.is_loaded:
        collection.load()

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k
    )
    return [
        {
            "id": hit.id,
            "score": hit.distance
        }
        for hit in results[0]
    ]
