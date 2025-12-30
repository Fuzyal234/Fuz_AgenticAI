# Pinecone Setup Guide for FUZ_AgenticAI

This guide will help you get started with Pinecone for your Agentic AI system.

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
./setup_pinecone.sh
```

This script will:
- Create a `.env` file from `.env.example`
- Install Pinecone CLI (if needed)
- Create a Python virtual environment (if needed)
- Install all dependencies
- Create your Pinecone index

### Option 2: Manual Setup

Follow these steps:

#### 1. Install Pinecone CLI

**macOS:**
```bash
brew tap pinecone-io/tap
brew install pinecone-io/tap/pinecone
```

**Other platforms:**
Download from [GitHub Releases](https://github.com/pinecone-io/cli/releases)

Verify installation:
```bash
pc version
```

#### 2. Get Your API Key

1. Sign up at [https://app.pinecone.io/](https://app.pinecone.io/)
2. Get your API key from the dashboard
3. Copy it for the next step

#### 3. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```bash
PINECONE_API_KEY=your_actual_api_key_here
OPENAI_API_KEY=your_openai_key_here
```

#### 4. Install Dependencies

Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install packages:
```bash
pip install -r requirements.txt
```

#### 5. Create Pinecone Index

Export your API key and create the index:
```bash
export PINECONE_API_KEY=your_api_key_here
pc index create -n fuz-agentic-ai -m cosine -c aws -r us-east-1 \
  --model llama-text-embed-v2 \
  --field_map text=content
```

Wait a few seconds for the index to be ready.

#### 6. Verify Setup

Test the connection:
```bash
python -c "from memory.pinecone_store import memory_store; print('✅ Pinecone connected!')"
```

## What Changed?

### Updated to New Pinecone API

Your code has been updated to use the **new Pinecone API** (2025) which includes:

1. **Integrated Embeddings**: Pinecone generates embeddings automatically - no need to create them manually
2. **Namespaces**: All operations now use namespaces for data isolation
3. **New SDK**: Using `pinecone` package instead of deprecated `pinecone-client`
4. **Better Search**: Includes reranking for improved results

### Key Differences

**Old API (deprecated):**
```python
# Manual embedding generation
embedding = llm_client.get_embedding(code)
index.upsert(vectors=[{"id": "...", "values": embedding, "metadata": {...}}])
index.query(vector=embedding, top_k=5)
```

**New API (current):**
```python
# Automatic embedding generation
index.upsert_records("namespace", [{"_id": "...", "content": code, ...}])
index.search("namespace", query={"inputs": {"text": query}, "top_k": 5})
```

## Usage Examples

### Storing Code

```python
from memory.pinecone_store import memory_store

# Store code in default namespace
memory_store.store_code(
    code="def hello(): print('world')",
    file_path="src/main.py",
    commit_hash="abc123"
)

# Store in custom namespace
memory_store.store_code(
    code="...",
    file_path="...",
    namespace="project-specific"
)
```

### Searching

```python
# Search for similar code
results = memory_store.search_similar(
    query="function that prints hello",
    top_k=5
)

# Get formatted context
context = memory_store.get_relevant_context(
    query="authentication logic",
    max_results=10
)
```

### Storing Decisions

```python
memory_store.store_decision(
    decision="Use FastAPI for API layer",
    context="Need REST API with async support",
    agent="planner"
)
```

## Important Notes

1. **Namespaces are Mandatory**: All operations require a namespace. The default is `"agentic-memory"`.

2. **Wait After Upserting**: If you upsert records and immediately search, wait 10+ seconds:
   ```python
   memory_store.store_code(...)
   results = memory_store.search_similar(..., wait_after_upsert=True)
   ```

3. **Field Names Must Match**: The `content` field name must match the `--field_map text=content` used when creating the index.

4. **Flat Metadata Only**: Metadata cannot contain nested objects - only flat key-value pairs.

## Troubleshooting

### Index Not Found Error

If you see "Index does not exist", create it:
```bash
pc index create -n fuz-agentic-ai -m cosine -c aws -r us-east-1 \
  --model llama-text-embed-v2 \
  --field_map text=content
```

### API Key Not Set

Make sure your `.env` file has:
```
PINECONE_API_KEY=your_actual_key_here
```

And that you're loading it (the code now uses `python-dotenv` automatically).

### Search Returns No Results

1. Make sure you've upserted data first
2. Wait 10+ seconds after upserting before searching
3. Check that you're using the correct namespace

## Next Steps

- Read the full Pinecone documentation: [https://docs.pinecone.io/](https://docs.pinecone.io/)
- Check the reference guides in `.agents/` directory
- Test your setup with the examples above

## Resources

- [Pinecone Dashboard](https://app.pinecone.io/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Python SDK Guide](.agents/PINECONE-python.md)
- [Quickstart Guide](.agents/PINECONE-quickstart.md)

