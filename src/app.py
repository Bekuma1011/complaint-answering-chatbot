import streamlit as st
import logging
from typing import List, Dict, Tuple
from rag_pipeline import RAGPipeline, create_rag_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotApp:
    """Main Streamlit application class for the complaint chatbot."""
    
    def __init__(self):
        self.rag_pipeline = None
        self.is_initialized = False
        self.logger = logger
        
        # Page configuration
        st.set_page_config(
            page_title="CrediTrust Complaint Chatbot", 
            page_icon="💬",
            layout="wide"
        )
    
    def initialize(self):
        """Initialize the RAG pipeline."""
        try:
            with st.spinner("Initializing AI models and vector store..."):
                self.rag_pipeline = create_rag_pipeline()
                self.is_initialized = True
                st.success(" System initialized successfully!")
                
        except Exception as e:
            st.error(f" Failed to initialize system: {str(e)}")
            self.logger.error(f"Initialization failed: {e}")
            return False
        
        return True
    
    def display_header(self):
        """Display the application header."""
        st.title("💬 CrediTrust Complaint Insights")
        st.markdown("""
        Ask questions about customer complaints across products like:
        - 💳 Credit Cards
        - 💰 Personal Loans  
        - 🛒 Buy Now Pay Later
        - 🏦 Savings Accounts
        - 💸 Money Transfers
        """)
        
        # Display system status
        status = self.rag_pipeline.get_pipeline_status() if self.rag_pipeline else {}
        if status.get("pipeline_ready"):
            st.sidebar.success("🟢 System Ready")
        else:
            st.sidebar.error("🔴 System Not Ready")
        if status:
            st.sidebar.json(status)
    
    def process_user_query(self, user_input: str) -> Tuple[str, List[str]]:
        """Process a user query through the RAG pipeline."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        try:
            with st.spinner("🤔 Thinking..."):
                answer, sources = self.rag_pipeline.process_query(user_input)
                return answer, sources
                
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    def display_answer(self, answer: str, sources: List[str]):
        """Display the generated answer and sources."""
        # Answer section
        st.subheader("🧠 Answer")
        st.write(answer)
        
        # Sources section
        st.subheader("📚 Sources")
        for i, src in enumerate(sources):
            with st.expander(f"Source {i+1}"):
                st.markdown(f"**Preview:**")
                st.markdown(f"> {src[:400]}...")
                if len(src) > 400:
                    st.markdown(f"**Full text:**")
                    st.text_area(f"Complete source {i+1}", src, height=150, key=f"source_{i}")
    
    def display_error(self, error_message: str):
        """Display error messages to the user."""
        st.error(f"❌ Error: {error_message}")
        st.info("💡 Please try rephrasing your question or contact support if the issue persists.")
    
    def run(self):
        """Main method to run the Streamlit application."""
        try:
            # Initialize if not already done
            if not self.is_initialized:
                if not self.initialize():
                    st.stop()
            
            # Display header after initialization to ensure correct status
            self.display_header()
            
                

            # Main chat interface
            with st.form(key="question_form"):
                user_input = st.text_input(
                    "Enter your question:", 
                    placeholder="e.g., Why are people unhappy with BNPL?",
                    help="Ask any question about customer complaints and their patterns"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    submit = st.form_submit_button("🚀 Ask", use_container_width=True)
                with col2:
                    clear = st.form_submit_button("🗑️ Clear", use_container_width=True)
            
            # Process form submission
            if submit and user_input.strip():
                try:
                    answer, sources = self.process_user_query(user_input)
                    self.display_answer(answer, sources)
                    
                except Exception as e:
                    self.display_error(str(e))
                    self.logger.error(f"Query processing failed: {e}")
            
            # Handle clear button
            if clear:
                st.experimental_rerun()
            
            # Additional features in sidebar
            self.display_sidebar_features()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            self.logger.error(f"Application error: {e}")
    
    def display_sidebar_features(self):
        """Display additional features in the sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 System Info")
        
        if self.is_initialized:
            # Model information
            st.sidebar.markdown("**Models:**")
            st.sidebar.markdown("- 🧠 Embedding: all-MiniLM-L6-v2")
            st.sidebar.markdown("- 🤖 LLM: Flan-T5-Base")
            
            # Performance metrics
            st.sidebar.markdown("**Performance:**")
            st.sidebar.markdown("- ⚡ Vector Search: ChromaDB")
            st.sidebar.markdown("- 📊 Chunk Size: 400 tokens")
            st.sidebar.markdown("- 🔍 Overlap: 50 tokens")
        
        # Help section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 Tips")
        st.sidebar.markdown("""
        - Be specific in your questions
        - Ask about trends or patterns
        - Request examples when relevant
        - Use product names for better results
        """)
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ About")
        st.sidebar.markdown("""
        This chatbot uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on real customer complaint data.
        """)
    
    def cleanup(self):
        """Clean up resources when the app is closed."""
        try:
            if self.rag_pipeline:
                self.rag_pipeline.cleanup()
            self.logger.info("Application cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def main():
    """Main function to run the chatbot application."""
    app = ChatbotApp()
    
    try:
        app.run()
    except KeyboardInterrupt:
        app.cleanup()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        app.cleanup()


if __name__ == "__main__":
    main()
