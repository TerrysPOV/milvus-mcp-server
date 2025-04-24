from server import (
    ingest_file_to_milvus,
    query_document_summary,
    create_collection_with_doc_id,
    list_collections,
    health_check
)

print("Collections:", list_collections())
print("Health:", health_check())

# Create or reuse a collection
print(create_collection_with_doc_id("pdf_test_collection"))

# Ingest PDF
print(ingest_file_to_milvus("example_pdf.pdf", "pdf_test_collection", doc_id="pdf-doc-1"))

# Ingest DOCX
print(ingest_file_to_milvus("summary_docx.docx", "pdf_test_collection", doc_id="docx-doc-1"))

# Run query
print(query_document_summary("pdf_test_collection", "What is this document about?", top_k=5))
