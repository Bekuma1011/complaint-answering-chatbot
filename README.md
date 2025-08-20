# Complaint Answering Chatbot – RAG Pipeline with OOP 

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering customer complaints using a large language model (LLM) and vector-based document retrieval. The system is built with **Object-Oriented Programming (OOP)** principles and includes comprehensive **CI/CD pipelines** for automated quality checks and deployment.

## 🏗️ Architecture Overview

The system follows a modular, class-based architecture:

```
src/
├── models.py          # Core model classes (EmbeddingModel, LLMModel, VectorStore)
├── rag_pipeline.py    # Main RAG pipeline orchestration
├── chunk_embed_index.py # Data processing and indexing
└── app.py            # Streamlit web interface
```

### 🔧 Core Classes

- **`BaseModel`**: Abstract base class for all AI models
- **`EmbeddingModel`**: Handles text embedding operations
- **`LLMModel`**: Manages language model operations
- **`VectorStore`**: Vector database operations
- **`DataProcessor`**: Data loading and validation
- **`TextChunker`**: Text chunking operations
- **`RAGPipeline`**: Main pipeline orchestration
- **`ChatbotApp`**: Streamlit application interface

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/complaint-answering-chatbot.git
   cd complaint-answering-chatbot
   ```

2. **Create virtual environment**
   ```bash
   make venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   make install
   ```

4. **Initialize the system**
   ```bash
   make db-init
   ```

5. **Run the application**
   ```bash
   make run
   ```

The chatbot will be available at `http://localhost:8501`

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Start all services
make docker-run

# View logs
make docker-logs

# Stop services
make docker-stop
```

### Manual Docker Build

```bash
# Build image
make build

# Run container
docker run -p 8501:8501 complaint-chatbot:latest
```

## 🧪 Testing & Quality Assurance

### Automated Testing

```bash
# Run all tests
make test

# Run specific test types
make unit          # Unit tests only
make integration   # Integration tests only
make performance   # Performance tests only

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality Checks

```bash
# Run all quality checks
make lint

# Format code
make format

# Security scanning
make security

# Pre-commit hooks
make pre-commit
```

### Full CI Pipeline (Local)

```bash
# Run complete local CI pipeline
make ci-local
```

## 🔄 CI/CD Pipeline

The project includes comprehensive CI/CD automation:

### GitHub Actions Workflow

- **Code Quality**: Black, Flake8, MyPy, isort
- **Testing**: Unit, integration, and performance tests
- **Security**: Bandit and Safety scans
- **Documentation**: Auto-generated API docs
- **Deployment**: Docker build and push
- **Monitoring**: Prometheus and Grafana

### Pre-commit Hooks

Automated quality checks before each commit:
- Code formatting (Black)
- Import sorting (isort)
- Linting (Flake8)
- Type checking (MyPy)
- Security scanning (Bandit)

### Quality Gates

- **Code Coverage**: Minimum 80%
- **Linting**: Zero violations
- **Type Checking**: No critical errors
- **Security**: No high-risk vulnerabilities

## 📊 Monitoring & Observability

### Built-in Monitoring

- **Health Checks**: Application and service health monitoring
- **Metrics**: Performance and usage metrics
- **Logging**: Structured logging with different levels
- **Error Tracking**: Comprehensive error handling and reporting

### Optional Services

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Redis**: Caching layer
- **Nginx**: Reverse proxy and load balancing

## 🛠️ Development Workflow

### 1. Setup Development Environment



### 2. Make Changes

- Follow OOP principles
- Add comprehensive tests
- Update documentation

### 3. Quality Checks

```bash
make pre-commit  # Automatic checks
make lint        # Manual checks
make test        # Run tests
```

### 4. Commit & Push

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature-branch
```

## 📁 Project Structure

```
complaint-answering-chatbot/
├── .github/                    # GitHub Actions workflows
├── src/                        # Source code
│   ├── models.py              # Core model classes
│   ├── rag_pipeline.py        # RAG pipeline
│   ├── chunk_embed_index.py   # Data processing
│   └── app.py                 # Streamlit app
├── tests/                      # Test suite
│   ├── test_models.py         # Model tests
│   ├── test_rag_pipeline.py   # Pipeline tests
│   └── test_app.py            # App tests
├── data/                       # Data files
├── docs/                       # Generated documentation
├── .pre-commit-config.yaml    # Pre-commit hooks
├── docker-compose.yml          # Docker services
├── Dockerfile                  # Multi-stage Docker build
├── Makefile                    # Development commands
├── pytest.ini                 # Test configuration
└── requirements.txt            # Dependencies
```

## 🔧 Configuration

### Environment Variables

```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=false

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base

# Vector Store Configuration
VECTOR_STORE_PATH=../data/vector_store
COLLECTION_NAME=complaints
```

### Model Parameters

- **Chunk Size**: 400 tokens
- **Chunk Overlap**: 50 tokens
- **Embedding Model**: all-MiniLM-L6-v2
- **LLM**: Flan-T5-Base
- **Vector Database**: ChromaDB










