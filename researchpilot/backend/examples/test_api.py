"""
Example Usage Script - ResearchPilot AI Agent

This script demonstrates how to use the ResearchPilot API programmatically.
Run this after the FastAPI server is running.

Usage:
    python examples/test_api.py
"""

import requests
import json
import time
from typing import Dict, Any


class ResearchPilotClient:
    """
    Python client for ResearchPilot API.
    
    Demonstrates usage patterns for:
    - Document ingestion
    - Semantic search
    - Context creation
    - Chat interactions
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the ResearchPilot API
        """
        self.base_url = base_url
        self.api_v1 = f"{base_url}/api/v1"
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def ingest_documents(self) -> Dict[str, Any]:
        """Ingest documents from the data directory."""
        response = self.session.post(f"{self.api_v1}/papers/ingest")
        return response.json()
    
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar documents."""
        params = {"query": query, "top_k": top_k}
        response = self.session.get(f"{self.api_v1}/papers/search", params=params)
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        response = self.session.get(f"{self.api_v1}/papers/stats")
        return response.json()
    
    def chat(self, query: str, use_context: bool = True, top_k: int = 5) -> Dict[str, Any]:
        """Chat with context."""
        data = {
            "query": query,
            "use_context": use_context,
            "top_k": top_k
        }
        response = self.session.post(f"{self.api_v1}/chat/chat", json=data)
        return response.json()
    
    def create_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Create research context."""
        data = {
            "query": query,
            "top_k": top_k
        }
        response = self.session.post(f"{self.api_v1}/chat/context", json=data)
        return response.json()
    
    def clear_collection(self) -> Dict[str, Any]:
        """Clear the vector collection."""
        response = self.session.post(f"{self.api_v1}/papers/clear")
        return response.json()


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_response(response: Dict[str, Any], indent: int = 2):
    """Pretty print API response."""
    print(json.dumps(response, indent=indent))


def main():
    """Run example API calls."""
    
    print_section("ResearchPilot AI Agent - API Examples")
    
    # Initialize client
    client = ResearchPilotClient()
    
    # Example 1: Health Check
    print_section("1. Health Check")
    try:
        health = client.health_check()
        print_response(health)
        print("✓ API is healthy")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure the FastAPI server is running (python main.py)")
        return
    
    # Example 2: Ingest Documents
    print_section("2. Ingest Documents from data/ Directory")
    try:
        result = client.ingest_documents()
        print_response(result)
        
        if result.get("status") == "success":
            print(f"✓ Successfully ingested {result['documents_ingested']} document chunks")
        elif result.get("status") == "warning":
            print("⚠ No PDF files found in data/ directory")
            print("   Please add PDF files to: researchpilot/backend/data/")
        else:
            print("✗ Ingestion failed")
    
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 3: Get Collection Stats
    print_section("3. Get Collection Statistics")
    try:
        stats = client.get_stats()
        print_response(stats)
        
        if stats.get("status") == "success":
            data = stats.get("data", {})
            print(f"✓ Collection contains {data.get('document_count', 0)} documents")
    
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 4: Search Documents
    print_section("4. Search for Similar Documents")
    
    test_queries = [
        "artificial intelligence",
        "machine learning",
        "neural networks"
    ]
    
    for query in test_queries:
        try:
            print(f"\nSearching for: '{query}'")
            results = client.search(query, top_k=3)
            
            if results.get("status") == "success":
                count = results.get("results_count", 0)
                print(f"✓ Found {count} relevant documents")
                
                if count > 0:
                    for result in results.get("results", [])[:2]:  # Show top 2
                        print(f"\n  Rank {result['rank']}: Similarity {result['similarity']:.4f}")
                        doc = result['document'][:100] + "..."
                        print(f"  {doc}")
                else:
                    print("  No documents found (add PDFs to data/ directory)")
            else:
                print(f"✗ Search failed")
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Example 5: Chat with Context
    print_section("5. Chat with Research Context")
    
    chat_query = "What is deep learning?"
    try:
        print(f"Query: {chat_query}")
        response = client.chat(chat_query, use_context=True, top_k=3)
        
        if response.get("status") == "success":
            print(f"✓ Retrieved {response['context_documents']} documents")
            print(f"  Message: {response['message']}")
            print(f"  Note: {response['note']}")
            
            # Show context preview (first 500 chars)
            context = response.get("context", "")
            if context:
                print(f"\n  Context Preview:\n  {context[:500]}...")
        else:
            print("✗ Chat failed")
    
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 6: Create Research Context
    print_section("6. Create Research Context")
    
    context_query = "explainability in AI systems"
    try:
        print(f"Query: {context_query}")
        response = client.create_context(context_query, top_k=2)
        
        if response.get("status") == "success":
            print(f"✓ Created context with {response['documents_retrieved']} documents")
            context = response.get("context", "")
            if context:
                print(f"\n  First 300 characters of context:")
                print(f"  {context[:300]}...")
        else:
            print("✗ Context creation failed")
    
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Summary
    print_section("Summary")
    print("""
✓ Successfully demonstrated ResearchPilot API

Next Steps:
1. Add PDF files to: researchpilot/backend/data/
2. Run: POST /api/v1/papers/ingest
3. Use: GET /api/v1/papers/search?query=<your_query>
4. Try: POST /api/v1/chat/chat with context

Interactive Testing:
- Visit: http://localhost:8000/api/docs
- Use the Swagger UI to test endpoints
- View full documentation at README.md
    """)


if __name__ == "__main__":
    main()
