"""
Document Loader Module
Handles PDF loading, text extraction, and document chunking for the research system.
"""

import os
import logging
from typing import List, Tuple
from pathlib import Path
from pypdf import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Handles loading and processing PDF documents.
    
    Features:
    - Load PDF files from specified directory
    - Extract text from PDFs
    - Chunk documents into manageable pieces
    - Preserve metadata for chunks
    """
    
    def __init__(
        self, 
        data_dir: str = "./data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize DocumentLoader.
        
        Args:
            data_dir: Directory containing PDF files
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentLoader initialized with data_dir: {self.data_dir}")
    
    def load_pdf(self, file_path: str) -> str:
        """
        Load and extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF reading fails
        """
        pdf_path = Path(file_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            text = ""
            pdf_reader = PdfReader(pdf_path)
            logger.info(f"Loading PDF: {pdf_path.name} ({len(pdf_reader.pages)} pages)")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num} ---\n{page_text}"
            
            logger.info(f"Successfully extracted text from {pdf_path.name}")
            return text
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Tuple[str, dict]]:
        """
        Split text into overlapping chunks with metadata preservation.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of tuples (chunk_text, chunk_metadata)
        """
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        metadata = metadata or {}
        
        # Calculate number of chunks needed
        if len(text) <= self.chunk_size:
            chunks.append((text, {**metadata, "chunk_index": 0}))
            return chunks
        
        # Create overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Avoid cutting mid-word
            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space > self.chunk_size // 2:
                    end = start + last_space
                    chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunk_metadata = {**metadata, "chunk_index": chunk_index}
                chunks.append((chunk.strip(), chunk_metadata))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def load_documents_from_directory(self) -> List[Tuple[str, dict]]:
        """
        Load all PDF documents from the data directory and chunk them.
        
        Returns:
            List of (chunk_text, metadata) tuples
        """
        all_chunks = []
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                # Load PDF
                text = self.load_pdf(str(pdf_file))
                
                # Create metadata for the document
                metadata = {
                    "source": pdf_file.name,
                    "file_path": str(pdf_file),
                    "document_type": "pdf"
                }
                
                # Chunk the document
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
                
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks created")
            
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def get_sample_documents(self) -> List[Tuple[str, dict]]:
        """
        Get documents from the data directory.
        
        Returns:
            List of (chunk_text, metadata) tuples
        """
        return self.load_documents_from_directory()
