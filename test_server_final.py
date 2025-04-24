
from server import (
    create_collection_with_text,
    embed_and_insert_texts,
    search_with_metadata,
    list_collections,
    health_check
)

# Step 1: Create collection with text
print(create_collection_with_text("test_text_collection"))

# Step 2: Insert some raw text data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Milvus is an open-source vector database.",
    "Claude is a conversational AI built by Anthropic.",
    "Vector search enables semantic similarity.",
    "FastMCP integrates Claude with local tools."
]
print(embed_and_insert_texts("test_text_collection", texts))

# Step 3: Perform a search with one of the texts (simulated query vector)
query_text = "Conversational AI systems like Claude"
query_vector = [float(ord(c)) / 255.0 for c in query_text]
query_vector = query_vector[:1536] + [0.0] * (1536 - len(query_vector))

print("Search results with metadata:")
results = search_with_metadata("test_text_collection", query_vector)
for res in results:
    print(res)

# Step 4: Sanity checks
print("Collections:", list_collections())
print("Health check:", health_check())
