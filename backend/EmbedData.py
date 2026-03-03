import torch
import pytesseract
from PIL import Image
import ollama
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import uuid

class DocumentIngestor:
    def __init__(self, ocr_model: str = "glm-ocr"):
        self.ocr_model = ocr_model

    def perform_ocr(self, image_path: str) -> str:
        if torch.cuda.is_available():
            try:
                response = ollama.generate(
                    model=self.ocr_model,
                    prompt="Extract all text from this image exactly as it appears. Do not add any additional comments or formatting.",
                    images=[image_path]
                )
                return response.get("response", "")
            except Exception as e:
                print(f"Ollama OCR failed: {e}. Falling back to pytesseract.")
                return self._pytesseract_ocr(image_path)
        else:
            return self._pytesseract_ocr(image_path)
            
    def _pytesseract_ocr(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"PyTesseract OCR failed: {e}")
            return ""


class EmbeddingPipeline:
    """Handle text chunking and embedding generation."""
    
    def __init__(self, embedding_model: str = "nomic-embed-text", chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize embedding pipeline.
        
        Args:
            embedding_model: Ollama embedding model name
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str, source: str) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            source: Source file name or identifier
        
        Returns:
            List of dicts with 'text', 'chunk_id', 'doc_id', 'source', 'created_at'
        """
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Skip empty chunks
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': f"{doc_id}_chunk_{chunk_num}",
                    'doc_id': doc_id,
                    'source': source,
                    'created_at': datetime.utcnow().isoformat()
                })
                chunk_num += 1
            
            start += self.chunk_size - self.chunk_overlap
        
        print(f"Created {len(chunks)} chunks from {source}")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts using Ollama.
        
        Args:
            texts: List of text strings
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = []
        
        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                embedding = response.get("embedding")
                if embedding:
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * 768)
        
        return np.array(embeddings, dtype=np.float32)
    
    def process_document(self, text: str, source: str) -> Tuple[List[Dict], np.ndarray]:
        """
        Full pipeline: chunk text and generate embeddings.
        
        Args:
            text: Document text
            source: Source file name
        
        Returns:
            Tuple of (chunks, embeddings_array)
        """
        doc_id = str(uuid.uuid4())
        chunks = self.chunk_text(text, doc_id, source)
        
        if not chunks:
            return [], np.array([], dtype=np.float32)
        
        texts_to_embed = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts_to_embed)
        
        return chunks, embeddings
