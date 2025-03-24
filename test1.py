import os
from ingest import create_or_load_faiss  # Import from your existing project

def get_faiss_document_count(vector_store):
    """
    Returns the number of stored document chunks in FAISS.
    """
    if vector_store is None:
        print("⚠️ No FAISS index found.")
        return 0
    return vector_store.index.ntotal  # ✅ Returns total stored vectors (chunks)

def get_faiss_docstore_count(vector_store):
    """
    Returns the number of stored documents in FAISS docstore.
    """
    if vector_store is None:
        print("⚠️ No FAISS index found.")
        return 0

    # ✅ Safely access docstore
    try:
        return len(vector_store.docstore._dict)  # ✅ Works for older versions
    except AttributeError:
        return len(vector_store.docstore)  # ✅ Works for newer versions

def print_faiss_stored_docs(vector_store):
    """
    Prints the sources of all stored documents.
    """
    if vector_store is None:
        print("⚠️ No FAISS index found.")
        return

    print("\n📄 Stored Documents in FAISS:")
    try:
        # ✅ Check for `_dict` first
        stored_docs = vector_store.docstore._dict
    except AttributeError:
        stored_docs = vector_store.docstore  # ✅ Fallback for newer versions

    for idx, (doc_id, doc) in enumerate(stored_docs.items()):
        print(f"📌 Document {idx+1}: {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    print("🔍 Loading FAISS Index...")
    vector_store = create_or_load_faiss()  # Load existing FAISS index

    doc_count = get_faiss_document_count(vector_store)
    docstore_count = get_faiss_docstore_count(vector_store)

    print(f"\n✅ Total document chunks in FAISS: {doc_count}")
    print(f"✅ Total unique documents in FAISS docstore: {docstore_count}")

    print_faiss_stored_docs(vector_store)  # Print stored document details
