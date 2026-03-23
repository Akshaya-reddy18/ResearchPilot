"""
Prepare the vector database by ingesting all PDFs in the data directory.
Run this script inside the backend virtual environment.
"""
import os
import sys

# Ensure backend root is on sys.path so `utils` imports resolve when running as a script
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BACKEND_DIR)

from utils.document_loader import DocumentLoader
from routers import papers


def main():
    # load .env if exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Use the exact same absolute paths as main.py
    db_path = os.path.abspath(os.path.join(BACKEND_DIR, "vector_db"))

    # Resolve data_dir to an absolute path as well
    env_data_dir = os.getenv("DATA_DIR")
    if env_data_dir:
        if not os.path.isabs(env_data_dir):
            data_dir = os.path.abspath(os.path.join(BACKEND_DIR, env_data_dir))
        else:
            data_dir = env_data_dir
    else:
        data_dir = os.path.abspath(os.path.join(BACKEND_DIR, "data"))

    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "research_papers")

    print(f"Using data_dir={data_dir}, db_path={db_path}, collection={collection_name}")

    # Initialize the papers router which will create the shared VectorStore
    papers.initialize_papers_router(db_path=db_path, collection_name=collection_name, data_dir=data_dir)

    loader = papers.document_loader
    store = papers.vector_store

    documents = loader.load_documents_from_directory()
    print(f"Found {len(documents)} document chunks to ingest")

    if not documents:
        print("No documents to ingest. Add PDFs to the data directory and try again.")
        return

    result = store.ingest_documents(documents)
    print("Ingestion result:", result)


if __name__ == "__main__":
    main()
