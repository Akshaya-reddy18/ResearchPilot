"""
Vector Store Module
Handles vector database operations using ChromaDB with sentence-transformers embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
# use PersistentClient for disk persistence
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector Store manager using ChromaDB.
    
    Features:
    - Initialize and manage ChromaDB collections
    - Generate embeddings using sentence-transformers
    - Store document chunks with metadata
    - Query similar documents using semantic search
    - Persistent storage
    """
    
    def __init__(
        self,
        db_path: str,
        collection_name: str = "research_papers",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize VectorStore.
        
        Args:
            db_path: Path to ChromaDB persistent storage
            collection_name: Name of the collection to work with
            embedding_model: Name of the sentence-transformer model
        """
        # enforce absolute resolved path for DB
        self.db_path = Path(db_path).resolve()
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        
        # ✅ Use PersistentClient to ensure on-disk persistence
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        logger.info(f"[VectorStore] Using DB Path: {self.db_path}")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{collection_name}' ready")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Embedding model loaded successfully")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Newer SentenceTransformer.encode may return a numpy array and
            # does not accept `convert_to_list`. Convert to list explicitly.
            embeddings = self.embedding_model.encode(texts)
            try:
                # If it's a numpy array
                embeddings_list = embeddings.tolist()
            except Exception:
                embeddings_list = list(embeddings)

            logger.info(f"Generated embeddings for {len(texts)} text(s)")
            return embeddings_list
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def ingest_documents(
        self,
        documents: List[tuple[str, dict]]
    ) -> Dict[str, Any]:
        """
        Ingest documents into the vector store.
        
        Args:
            documents: List of (text, metadata) tuples from DocumentLoader
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return {"status": "failed", "message": "No documents provided", "count": 0}
        
        try:
            texts = []
            metadatas = []
            ids = []
            
            # Prepare documents for ingestion
            for idx, (text, metadata) in enumerate(documents):
                texts.append(text)
                metadatas.append(metadata)
                ids.append(f"doc_{idx}")
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.generate_embeddings(texts)
            
            # Add to collection
            logger.info(f"Adding {len(texts)} documents to collection...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully ingested {len(texts)} document chunks")
            return {
                "status": "success",
                "message": f"Ingested {len(texts)} document chunks",
                "count": len(texts)
            }
        
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return {
                "status": "failed",
                "message": str(e),
                "count": 0
            }
    
    def query_similar_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores
        """
        
        try:
            # Check if collection has documents
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("Collection is empty, no documents to query")
                return []
            
            # Generate query embedding (encode returns array; make plain list)
            query_embedding = self.embedding_model.encode([query])
            try:
                query_embedding = query_embedding.tolist()[0]
            except Exception:
                query_embedding = list(query_embedding)[0]
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection_count),
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results and results["documents"] and len(results["documents"]) > 0:
                for idx, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    )
                ):
                    # Convert distance to similarity (for cosine, 1 - distance)
                    similarity = 1 - distance
                    
                    formatted_results.append({
                        "rank": idx + 1,
                        "document": doc,
                        "metadata": metadata,
                        "similarity": round(similarity, 4),
                        "distance": round(distance, 4)
                    })
            
            logger.info(f"Query returned {len(formatted_results)} results")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
                "db_path": str(self.db_path)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
