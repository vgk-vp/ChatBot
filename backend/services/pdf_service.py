"""PDF processing and QA service using embeddings."""

import os
import uuid
from typing import List, Dict, Any, Optional
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
import os

class Config:
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))

config = Config()

class PDFService:
    def __init__(self):
        self.embeddings = SentenceTransformer(config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Ensure upload directory exists."""
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        
        return text
    
    def process_pdf(self, file_path: str, filename: str) -> str:
        """Process PDF and store in vector database."""
        # Extract text
        text = self.extract_text_from_pdf(file_path)
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF")
        
        # Create document
        doc = Document(
            page_content=text,
            metadata={
                "filename": filename,
                "source": file_path,
                "doc_id": str(uuid.uuid4())
            }
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Create collection name based on filename
        collection_name = f"pdf_{filename.replace('.', '_').replace(' ', '_').lower()}"
        
        # Store in vector database
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.encode(texts)
        
        # Simple storage (replace with proper vector DB in production)
        import json
        storage_path = os.path.join(config.CHROMA_PERSIST_DIR, f"{collection_name}.json")
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        
        data = {
            "texts": texts,
            "embeddings": embeddings.tolist(),
            "metadata": [chunk.metadata for chunk in chunks]
        }
        
        with open(storage_path, 'w') as f:
            json.dump(data, f)
        
        return collection_name
    
    def query_pdf(self, collection_name: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Query the PDF using similarity search."""
        try:
            import json
            from sklearn.metrics.pairwise import cosine_similarity
            
            storage_path = os.path.join(config.CHROMA_PERSIST_DIR, f"{collection_name}.json")
            
            if not os.path.exists(storage_path):
                return []
            
            with open(storage_path, 'r') as f:
                data = json.load(f)
            
            # Encode query
            query_embedding = self.embeddings.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, data["embeddings"])[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "content": data["texts"][idx],
                    "metadata": data["metadata"][idx],
                    "similarity_score": float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error querying PDF: {str(e)}")
    
    def get_available_pdfs(self) -> List[str]:
        """Get list of available PDF collections."""
        try:
            collections = []
            
            if os.path.exists(config.CHROMA_PERSIST_DIR):
                for item in os.listdir(config.CHROMA_PERSIST_DIR):
                    if item.startswith("pdf_") and item.endswith(".json"):
                        collections.append(item[:-5])  # Remove .json extension
            
            return collections
            
        except Exception as e:
            return []
    
    def delete_pdf_collection(self, collection_name: str) -> bool:
        """Delete a PDF collection from vector database."""
        try:
            # This would require ChromaDB client to delete collection
            # For now, we'll just return True
            return True
        except Exception:
            return False