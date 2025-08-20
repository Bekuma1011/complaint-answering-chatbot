from typing import List, Dict, Tuple, Optional
import logging
from models import EmbeddingModel, LLMModel, VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline class that orchestrates the entire process."""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 llm_model_name: str = "google/flan-t5-base",
                 vector_store_path: str = "../data/vector_store",
                 collection_name: str = "complaints"):
        
        # Initialize components
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.llm_model = LLMModel(llm_model_name)
        self.vector_store = VectorStore(vector_store_path, collection_name)
        
        # State tracking
        self.is_initialized = False
        self.logger = logger
    
    def initialize(self):
        """Initialize all components of the RAG pipeline."""
        try:
            self.logger.info("Initializing RAG pipeline...")
            
            # Load models
            self.embedding_model.load()
            self.llm_model.load()
            
            # Connect to vector store
            self.vector_store.connect()
            
            self.is_initialized = True
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the pipeline is ready for use."""
        return (
            self.is_initialized
            and self.embedding_model.is_loaded()
            and self.llm_model.is_loaded()
            and self.vector_store.is_connected
        )
    
    def retrieve_chunks(self, query: str, k: int = 3) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant chunks from the vector store."""
        if not self.is_ready():
            raise RuntimeError("RAG pipeline not ready. Call initialize() first.")
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            
            # Retrieve chunks
            chunks, metadatas = self.vector_store.query(query_embedding, k)
            
            self.logger.info(f"Retrieved {len(chunks)} chunks for query")
            return chunks, metadatas
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise
    
    def build_prompt(self, context_chunks: List[str], question: str) -> str:
        """Build a prompt for the LLM using retrieved context."""
        context = "\n\n".join(context_chunks)
        
        prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate answer using the LLM."""
        if not self.is_ready():
            raise RuntimeError("RAG pipeline not ready. Call initialize() first.")
        
        try:
            answer = self.llm_model.generate(prompt, max_new_tokens, do_sample=False)
            self.logger.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            raise
    
    def process_query(self, question: str, k: int = 5) -> Tuple[str, List[str]]:
        """Process a complete query through the RAG pipeline."""
        if not self.is_ready():
            raise RuntimeError("RAG pipeline not ready. Call initialize() first.")
        
        try:
            self.logger.info(f"Processing query: {question}")
            
            # Retrieve relevant chunks
            chunks, metadatas = self.retrieve_chunks(question, k)
            
            # Build prompt
            prompt = self.build_prompt(chunks, question)
            
            # Generate answer
            answer = self.generate_answer(prompt)
            
            self.logger.info("Query processed successfully")
            return answer, chunks[:2]  # Return first 2 chunks as sources
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get the status of all pipeline components."""
        return {
            "pipeline_initialized": self.is_initialized,
            "embedding_model_loaded": self.embedding_model.is_loaded(),
            "llm_model_loaded": self.llm_model.is_loaded(),
            "vector_store_connected": self.vector_store.is_connected,
            "pipeline_ready": self.is_ready()
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.vector_store.is_connected:
                self.vector_store.persist()
            self.logger.info("RAG pipeline cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


# Factory function for easy pipeline creation
def create_rag_pipeline(embedding_model_name: str = "all-MiniLM-L6-v2",
                       llm_model_name: str = "google/flan-t5-base",
                       vector_store_path: str = "../data/vector_store",
                       collection_name: str = "complaints") -> RAGPipeline:
    """Create and initialize a RAG pipeline."""
    pipeline = RAGPipeline(embedding_model_name, llm_model_name, vector_store_path, collection_name)
    pipeline.initialize()
    return pipeline






