import os
from routers import papers, chat
from utils.research_agent import ResearchAgent
from main import DB_PATH

collection_name = os.getenv("CHROMA_COLLECTION_NAME", "research_papers")
data_dir = os.getenv("DATA_DIR", "./data")

def main():
    print('USING DB_PATH:', DB_PATH)

    # Initialize papers router (creates its VectorStore)
    papers.initialize_papers_router(DB_PATH, collection_name, data_dir)
    try:
        pap_count = papers.vector_store.collection.count()
    except Exception as e:
        pap_count = f'ERROR: {e}'
    print('[PAPERS ROUTER] COLLECTION COUNT:', pap_count)

    # Initialize chat router (creates ResearchAgent and sets chat.agent)
    chat.initialize_chat_router(DB_PATH, collection_name, None)
    try:
        chat_count = chat.agent.vector_store.collection.count() if chat.agent else 'NO_AGENT'
    except Exception as e:
        chat_count = f'ERROR: {e}'
    print('[CHAT ROUTER] COLLECTION COUNT:', chat_count)

    # Also show ResearchAgent constructed directly (should match)
    ra = ResearchAgent(DB_PATH, collection_name)
    try:
        ra_count = ra.vector_store.collection.count()
    except Exception as e:
        ra_count = f'ERROR: {e}'
    print('[ResearchAgent] COLLECTION COUNT:', ra_count)

if __name__ == '__main__':
    main()
