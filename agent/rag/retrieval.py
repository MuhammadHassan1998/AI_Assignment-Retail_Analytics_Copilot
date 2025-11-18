"""RAG retrieval system using TF-IDF for document chunking and search."""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    
    def __init__(self, chunk_id: str, content: str, source: str, chunk_index: int):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.chunk_index = chunk_index
    
    def __repr__(self):
        return f"DocumentChunk(id={self.chunk_id}, source={self.source}, index={self.chunk_index})"


class TFIDFRetriever:
    """TF-IDF based retriever for document chunks."""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.vectors = None
        self._load_documents()
    
    def _load_documents(self):
        """Load and chunk all documents from the docs directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        
        for doc_file in self.docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into paragraphs/sections
            sections = self._chunk_document(content)
            
            for idx, section in enumerate(sections):
                if section.strip():
                    source_name = doc_file.stem
                    chunk_id = f"{source_name}::chunk{idx}"
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=section.strip(),
                        source=source_name,
                        chunk_index=idx
                    )
                    self.chunks.append(chunk)
        
        if not self.chunks:
            raise ValueError("No document chunks found!")
        
        # Fit vectorizer on all chunks
        texts = [chunk.content for chunk in self.chunks]
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def _chunk_document(self, content: str) -> List[str]:
        """Split document into chunks (paragraphs or sections)."""
        # Split by double newlines first (paragraphs)
        paragraphs = re.split(r'\n\n+', content)
        chunks = []
        
        for para in paragraphs:
            # If paragraph is too long, split by single newlines
            if len(para) > 500:
                lines = para.split('\n')
                current_chunk = []
                for line in lines:
                    if len(' '.join(current_chunk)) + len(line) > 500 and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [line]
                    else:
                        current_chunk.append(line)
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
            else:
                chunks.append(para)
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-k most relevant chunks for a query."""
        if not self.chunks or self.vectors is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk:
        """Get a chunk by its ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise ValueError(f"Chunk not found: {chunk_id}")


def create_retriever(docs_dir: str = "docs") -> TFIDFRetriever:
    """Factory function to create a retriever."""
    return TFIDFRetriever(docs_dir)

