import os
from typing import Dict, Any, List
from utils.vector_store import VectorStore
from utils.llm_client import GroqClient


class ResearchAgent:
    def __init__(self, db_path: str, collection_name: str):
        """
        ResearchAgent requires an explicit `db_path` and `collection_name`.

        This enforces that the path is provided by `main.py` at startup
        and avoids accidental use of a relative or in-memory DB.
        """
        self.vector_store = VectorStore(db_path=db_path, collection_name=collection_name)

        # Production: do not emit debug prints here

        try:
            self.llm = GroqClient()
        except Exception:
            self.llm = None

    def analyze_topic(self, query: str, top_k: int = 5, use_context: bool = True) -> Dict[str, Any]:
        """
        Analyze a topic using optional context from the vector store.

        Args:
            query: User query string.
            top_k: Number of similar documents to retrieve from the vector store.
            use_context: If False, skip retrieval and call the LLM directly with the query.

        Returns:
            Dict with analysis, query and metadata about sources used.
        """
        results: List[Dict[str, Any]] = []

        # Production: avoid debug prints about DB/collection

        # Retrieve context if requested
        if use_context:
            results = self.vector_store.query_similar_documents(query, top_k=top_k)

        # Production: avoid printing retrieved results

        # If no context is found and use_context was requested, fall back to empty context
        if use_context and not results:
            context = ""
        else:
            context = "\n\n".join([r["document"] for r in results]) if results else ""

        # Build structured prompt
        prompt = f"""
You are a research intelligence assistant.

Using the research context below, generate a structured research analysis.

Research Context:
{context}

User Query:
{query}

Return output in this structure:

1. Executive Summary
2. Key Findings
3. Methodology Comparison
4. Research Gaps
5. Future Scope
"""

        # Call LLM if available, otherwise return prompt snippet as fallback
        if self.llm:
            response = self.llm.generate_response(prompt)
        else:
            response = "Groq client not configured. Set GROQ_API_KEY in .env to enable LLM responses.\n\n" + (prompt[:2000])

        # Normalize response to plain text analysis. Handle SDK response objects or plain strings.
        analysis = ""
        try:
            # groq SDK-style object
            if hasattr(response, "choices"):
                analysis = response.choices[0].message.content.strip()
            elif isinstance(response, dict) and "choices" in response:
                analysis = response["choices"][0]["message"]["content"].strip()
            elif isinstance(response, str):
                analysis = response.strip()
            else:
                analysis = str(response).strip()
        except Exception:
            analysis = str(response).strip()

        return {
            "query": query,
            "analysis": analysis,
            "source_chunks_used": len(results),
            "top_k": top_k,
            "model": "llama3-70b-8192"
        }
