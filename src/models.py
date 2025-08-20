from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models in the system."""
    
    @abstractmethod
    def load(self):
        """Load the model."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class EmbeddingModel(BaseModel):
    """Handles text embedding operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_model_loaded = False
    
    def load(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.is_model_loaded = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        return self.is_model_loaded
    
    def encode(self, text: str) -> List[float]:
        """Encode text to vector representation."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.encode(text).tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to vector representations."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.encode(texts).tolist()


class LLMModel(BaseModel):
    """Handles text generation using language models."""
    
    def __init__(self, model_name: str = "google/flan-t5-base", task: str = "text2text-generation"):
        self.model_name = model_name
        self.task = task
        self.pipeline = None
        self.is_model_loaded = False
    
    def load(self):
        """Load the language model."""
        try:
            logger.info(f"Loading LLM: {self.model_name}")
            self.pipeline = pipeline(self.task, model=self.model_name)
            self.is_model_loaded = True
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
    
    def is_loaded(self) -> bool:
        return self.is_model_loaded
    
    def generate(self, prompt: str, max_new_tokens: int = 100, do_sample: bool = False) -> str:
        """Generate text response."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            response = self.pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
            return response[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class VectorStore:
    """Manages vector database operations."""
    
    def __init__(self, path: str = "../data/vector_store", collection_name: str = "complaints"):
        self.path = path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.is_connected = False
    
    def connect(self):
        """Connect to the vector database."""
        try:
            logger.info(f"Connecting to vector store at: {self.path}")
            self.client = chromadb.PersistentClient(path=self.path)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self.is_connected = True
            logger.info("Vector store connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to vector store: {e}")
            raise
    
    def query(self, query_embedding: List[float], n_results: int = 3) -> Tuple[List[str], List[Dict]]:
        """Query the vector store."""
        if not self.is_connected:
            raise RuntimeError("Vector store not connected. Call connect() first.")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results["documents"][0], results["metadatas"][0]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def add_documents(self, ids: List[str], embeddings: List[List[float]], 
                     metadatas: List[Dict], documents: List[str]):
        """Add documents to the vector store."""
        if not self.is_connected:
            raise RuntimeError("Vector store not connected. Call connect() first.")
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Added {len(ids)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def persist(self):
        """Persist changes to disk."""
        if self.client:
            self.client.persist()
            logger.info("Vector store persisted to disk")


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            logger.info(f"Loading data from: {self.data_path}")
            self.data = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def get_data(self) -> pd.DataFrame:
        """Get loaded data."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.data
    
    def validate_data(self) -> bool:
        """Validate data integrity."""
        if self.data is None:
            return False
        
        required_columns = ["Complaint ID", "cleaned_Consumer_complaint_narrative", "Product"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("Data validation passed")
        return True


class TextChunker:
    """Handles text chunking operations."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = None
        self._initialize_splitter()
    
    def _initialize_splitter(self):
        """Initialize the text splitter."""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", " "]
            )
            logger.info("Text splitter initialized successfully")
        except ImportError:
            logger.error("LangChain not available. Install with: pip install langchain")
            raise
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if self.text_splitter is None:
            raise RuntimeError("Text splitter not initialized")
        
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Text split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Text splitting failed: {e}")
            raise
