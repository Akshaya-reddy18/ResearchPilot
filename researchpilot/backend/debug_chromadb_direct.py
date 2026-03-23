import os
from main import DB_PATH
import chromadb

def main():
    print('DB_PATH:', DB_PATH)
    print('DB exists:', os.path.exists(DB_PATH))
    if os.path.exists(DB_PATH):
        try:
            print('DB listing:', os.listdir(DB_PATH))
        except Exception as e:
            print('DB listing error:', e)

    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        print('Chroma client created:', type(client))
        try:
            cols = client.list_collections()
            print('Collections:', cols)
        except Exception as e:
            print('list_collections error:', e)
        try:
            coll = client.get_collection('research_papers')
            # Some chromadb versions expose count via coll.count() or len(coll.get())
            try:
                cnt = coll.count()
            except Exception:
                try:
                    cnt = len(coll.get(ids=[]).get('ids', []))
                except Exception:
                    cnt = 'unknown'
            print("'research_papers' collection count:", cnt)
        except Exception as e:
            print("get_collection error:", e)
    except Exception as e:
        print('Failed to create PersistentClient:', e)

if __name__ == '__main__':
    main()
