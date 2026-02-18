import os
from main import DB_PATH
from utils.vector_store import VectorStore

collection_name = os.getenv("CHROMA_COLLECTION_NAME", "research_papers")


def main():
    print('DB_PATH:', DB_PATH)
    print('DB exists:', os.path.exists(DB_PATH))
    if os.path.exists(DB_PATH):
        try:
            print('DB listing:', os.listdir(DB_PATH))
        except Exception as e:
            print('DB listing error:', e)

    try:
        vs = VectorStore(DB_PATH, collection_name)
        print('VectorStore db_path:', getattr(vs, 'db_path', 'N/A'))
        print('VectorStore collection name:', getattr(vs, 'collection_name', 'N/A'))
        try:
            cnt = vs.collection.count()
        except Exception as e:
            cnt = f'ERROR: {e}'
        print('VectorStore.collection.count():', cnt)
    except Exception as e:
        print('Failed to instantiate VectorStore:', e)

if __name__ == '__main__':
    main()
