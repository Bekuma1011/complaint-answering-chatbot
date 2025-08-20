# chunk_embed_index.py

import pandas as pd
import logging
from typing import List, Dict
from .models import DataProcessor, TextChunker, EmbeddingModel, VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIndexer:
    """Main class for chunking, embedding, and indexing data."""
    
    def __init__(self, 
                 data_path: str = "../data/processed/filtered_complaints12.csv",
                 vector_store_path: str = "../data/vector_store",
                 collection_name: str = "complaints",
                 chunk_size: int = 400,
                 chunk_overlap: int = 50):
        
        # Initialize components
        self.data_processor = DataProcessor(data_path)
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(vector_store_path, collection_name)
        
        # State tracking
        self.is_processed = False
        self.total_chunks = 0
        self.logger = logger
    
    def process_data(self):
        """Main method to process and index all data."""
        try:
            self.logger.info("Starting data processing and indexing...")
            
            # Load and validate data
            df = self.data_processor.load_data()
            if not self.data_processor.validate_data():
                raise ValueError("Data validation failed")
            
            # Load models and connect to vector store
            self.embedding_model.load()
            self.vector_store.connect()
            
            # Process each complaint
            self._process_complaints(df)
            
            # Persist to disk
            self.vector_store.persist()
            
            self.is_processed = True
            self.logger.info(f"✅ Processing complete. Total chunks indexed: {self.total_chunks}")
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
    
    def _process_complaints(self, df: pd.DataFrame):
        """Process individual complaints and create chunks."""
        doc_id = 0
        
        for idx, row in df.iterrows():
            try:
                complaint_text = row["cleaned_Consumer_complaint_narrative"]
                metadata = {
                    "complaint_id": str(row["Complaint ID"]),
                    "product": row["Product"]
                }
                
                # Split into chunks
                chunks = self.text_chunker.split_text(complaint_text)
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    self._process_chunk(chunk, metadata, i, doc_id)
                    doc_id += 1
                
                # Log progress every 1000 complaints
                if (idx + 1) % 1000 == 0:
                    self.logger.info(f"Processed {idx + 1}/{len(df)} complaints")
                    
            except Exception as e:
                self.logger.warning(f"Failed to process complaint {idx}: {e}")
                continue
    
    def _process_chunk(self, chunk: str, metadata: Dict, chunk_index: int, doc_id: int):
        """Process a single text chunk."""
        try:
            # Create embedding
            embedding = self.embedding_model.encode(chunk)
            
            # Prepare metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["text"] = chunk
            
            # Add to vector store
            self.vector_store.add_documents(
                ids=[f"doc-{doc_id}"],
                embeddings=[embedding],
                metadatas=[chunk_metadata],
                documents=[chunk]
            )
            
            self.total_chunks += 1
            
        except Exception as e:
            self.logger.error(f"Failed to process chunk {doc_id}: {e}")
            raise
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about the processing."""
        return {
            "is_processed": self.is_processed,
            "total_chunks": self.total_chunks,
            "data_loaded": self.data_processor.data is not None,
            "embedding_model_loaded": self.embedding_model.is_loaded(),
            "vector_store_connected": self.vector_store.is_connected()
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.vector_store.is_connected:
                self.vector_store.persist()
            self.logger.info("Data indexer cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def main():
    """Main function to run the indexing process."""
    try:
        # Create and run indexer
        indexer = DataIndexer()
        indexer.process_data()
        
        # Print final stats
        stats = indexer.get_processing_stats()
        logger.info(f"Final stats: {stats}")
        
        # Cleanup
        indexer.cleanup()
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


if __name__ == "__main__":
    main()