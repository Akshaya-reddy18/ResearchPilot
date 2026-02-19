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

        try:
            self.llm = GroqClient()
            # Get the model name from the client instance
            self.model_name = self.llm.model
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"GroqClient initialized successfully with model: {self.model_name}")
        except Exception as e:
            self.llm = None
            self.model_name = None
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"GroqClient initialization failed: {str(e)}. AI responses will be disabled.")

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

        # Retrieve context if requested
        if use_context:
            results = self.vector_store.query_similar_documents(query, top_k=top_k)

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

        # Call LLM if available, otherwise return helpful message
        if self.llm:
            response = self.llm.generate_response(prompt)
            # Extract analysis content from response
            if isinstance(response, str):
                analysis = response.strip()
            else:
                analysis = str(response).strip()
        else:
            # Return a structured response even without LLM
            if results:
                # We have context but no LLM - provide basic analysis
                analysis = f"""1. Executive Summary

Based on the {len(results)} relevant document chunks retrieved from your research library, this topic appears in your indexed documents. However, to generate a comprehensive AI-powered analysis, please configure the Groq API key.

2. Key Findings

{len(results)} relevant document chunks were found related to your query.

3. Methodology Comparison

To enable AI analysis, add GROQ_API_KEY to your backend .env file.

4. Research Gaps

Configure Groq API to unlock full research analysis capabilities.

5. Future Scope

Once GROQ_API_KEY is configured, the system will provide detailed AI-generated research analysis using the Llama 3.3 70B model."""
            else:
                # No context and no LLM
                analysis = f"""1. Executive Summary

No relevant documents were found in your research library for this topic. Please upload and ingest PDF documents first, then configure the Groq API key for AI-powered analysis.

2. Key Findings

- No documents indexed yet, or query doesn't match existing documents
- Groq API key not configured

3. Methodology Comparison

To get started:
1. Upload PDF documents via the Upload page
2. Ingest them into the vector database
3. Add GROQ_API_KEY to backend/.env file

4. Research Gaps

System is ready but needs:
- Document ingestion
- Groq API configuration

5. Future Scope

Once configured, the system will provide intelligent research analysis combining your documents with AI reasoning."""

        return {
            "query": query,
            "analysis": analysis,
            "source_chunks_used": len(results),
            "top_k": top_k,
            "model": self.model_name
        }
