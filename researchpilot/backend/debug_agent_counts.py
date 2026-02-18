from main import DB_PATH
from utils.vector_store import VectorStore
from utils.research_agent import ResearchAgent
import os

collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'research_papers')

print('DB_PATH:', DB_PATH)
vs = VectorStore(DB_PATH, collection_name)
try:
    print('[Direct VectorStore] collection count:', vs.collection.count())
except Exception as e:
    print('[Direct VectorStore] count error:', e)

ra = ResearchAgent(DB_PATH, collection_name)
try:
    print('[ResearchAgent] vector_store.collection.count():', ra.vector_store.collection.count())
except Exception as e:
    print('[ResearchAgent] count error:', e)
