# PineconeStore Documentation

## Overview

The `pinecone_store.py` file implements a **Pinecone vector store** for long-term memory storage in an agentic AI system. It uses Pinecone's new API with integrated embeddings and namespaces to store, retrieve, and search various types of data including code, decisions, error patterns, reasoning traces, and execution plans.

## Purpose

This module provides a persistent memory layer that allows the AI system to:
- Store and retrieve code snippets with semantic search
- Remember architectural and agent decisions
- Learn from error patterns and their fixes
- Store reasoning traces for future reference
- Save execution plans for similar tasks

## Key Features

- **Integrated Embeddings**: Uses Pinecone's built-in embedding generation (no need to manually create embeddings)
- **Namespace Isolation**: Supports multiple namespaces for data organization and isolation
- **Semantic Search**: Performs semantic similarity search with reranking for better results
- **Multiple Data Types**: Handles 5 different types of stored data (code, decisions, errors, reasoning, plans)
- **Automatic Index Management**: Validates index existence and provides setup guidance

## Class: `PineconeStore`

### Initialization

The class requires:
- `PINECONE_API_KEY`: API key from environment variables (via `config.settings`)
- `PINECONE_INDEX_NAME`: Name of the Pinecone index to use

On initialization, it:
1. Validates the API key exists
2. Connects to Pinecone
3. Ensures the index exists (raises error with setup instructions if not)
4. Creates a reference to the index

### Default Namespace

- Default namespace: `"agentic-memory"`
- All operations can use custom namespaces for data isolation

---

## Methods

### Storage Methods

#### `store_code(code, file_path, commit_hash=None, metadata=None, namespace=None)`

Stores code snippets in the vector database.

**Parameters:**
- `code`: Code content to store (truncated to 1000 chars)
- `file_path`: Path to the source file
- `commit_hash`: Optional Git commit hash
- `metadata`: Additional metadata dictionary (flat structure only)
- `namespace`: Optional namespace (defaults to `DEFAULT_NAMESPACE`)

**Stored Metadata:**
- `type`: "code"
- `file_path`: The file path
- `commit_hash`: If provided
- Any additional flat metadata fields

---

#### `store_decision(decision, context, agent, metadata=None, namespace=None)`

Stores architectural decisions or agent decisions.

**Parameters:**
- `decision`: The decision text
- `context`: Context surrounding the decision
- `agent`: Which agent made the decision
- `metadata`: Additional metadata
- `namespace`: Optional namespace

**Stored Metadata:**
- `type`: "decision"
- `agent`: Agent name
- `decision`: Decision text (truncated to 500 chars)
- `context`: Context text (truncated to 500 chars)

---

#### `store_error_pattern(error, fix, metadata=None, namespace=None)`

Stores error patterns and their corresponding fixes for future reference.

**Parameters:**
- `error`: Error message or pattern
- `fix`: The solution or fix
- `metadata`: Additional metadata
- `namespace`: Optional namespace

**Stored Metadata:**
- `type`: "error_pattern"
- `error`: Error text (truncated to 500 chars)
- `fix`: Fix text (truncated to 500 chars)

---

#### `store_reasoning_trace(problem, reasoning_steps, conclusion, confidence, reasoning_type="general", metadata=None, namespace=None)`

Stores reasoning traces from the LRM (Long-term Reasoning Memory) agent.

**Parameters:**
- `problem`: The problem being reasoned about
- `reasoning_steps`: List of reasoning step dictionaries
- `conclusion`: Final conclusion
- `confidence`: Confidence level (0.0-1.0)
- `reasoning_type`: Type of reasoning (general, architectural, debugging, planning)
- `metadata`: Additional metadata
- `namespace`: Optional namespace

**Stored Metadata:**
- `type`: "reasoning"
- `reasoning_type`: Type of reasoning
- `problem`: Problem text (truncated to 500 chars)
- `conclusion`: Conclusion text (truncated to 500 chars)
- `confidence`: Confidence score
- `num_steps`: Number of reasoning steps

---

#### `store_plan(user_request, plan, metadata=None, namespace=None)`

Stores execution plans. This method stores the plan as a whole AND each individual step separately for better searchability.

**Parameters:**
- `user_request`: Original user request
- `plan`: Dictionary containing:
  - `steps`: List of plan steps
  - `understanding`: System's understanding
  - `estimated_complexity`: Complexity level
  - `risks`: List of identified risks
- `metadata`: Additional metadata
- `namespace`: Optional namespace

**Stored Metadata (Plan):**
- `type`: "plan"
- `user_request`: Request text (truncated to 500 chars)
- `understanding`: Understanding text (truncated to 500 chars)
- `complexity`: Complexity level
- `num_steps`: Number of steps
- `risks`: Risk list (truncated to 500 chars)

**Stored Metadata (Each Step):**
- `type`: "plan_step"
- `user_request`: Request text
- `step_number`: Step index
- `agent`: Assigned agent
- `action`: Action description
- `files`: Files involved

---

### Search Methods

#### `search_similar(query, top_k=5, filter_dict=None, namespace=None, wait_after_upsert=False)`

Performs semantic similarity search with reranking.

**Parameters:**
- `query`: Search query text
- `top_k`: Number of results to return (default: 5)
- `filter_dict`: Optional metadata filter for filtering results
- `namespace`: Namespace to search in
- `wait_after_upsert`: If True, waits 10 seconds before searching (use after upserting)

**Returns:**
List of dictionaries with:
- `id`: Record ID
- `score`: Similarity score
- `metadata`: All metadata fields including content

**Features:**
- Uses semantic search with reranking (bge-reranker-v2-m3 model)
- Fetches 2x candidates initially, then reranks to top_k
- Automatically generates embeddings for the query

---

#### `get_relevant_context(query, max_results=10, namespace=None) -> str`

Retrieves formatted context strings from search results.

**Parameters:**
- `query`: Search query
- `max_results`: Maximum number of results (default: 10)
- `namespace`: Namespace to search

**Returns:**
Formatted string with context sections separated by `---`. Each section is formatted based on the record type:
- **Code**: Shows file path and code snippet
- **Decision**: Shows decision and context
- **Error Pattern**: Shows error and fix
- **Reasoning**: Shows problem, conclusion, and confidence
- **Plan**: Shows request, understanding, and step count
- **Plan Step**: Shows step number, action, agent, and files

---

### Helper Methods

#### `_generate_id(content, metadata) -> str`

Generates a unique MD5 hash ID for records based on content and metadata. This ensures duplicate content doesn't create multiple records.

#### `_ensure_index()`

Validates that the required Pinecone index exists. If not, raises an error with CLI instructions for creating it.

---

## Technical Details

### Index Requirements

The Pinecone index must be created with:
- **Metric**: cosine similarity
- **Embedding Model**: llama-text-embed-v2
- **Field Map**: `text=content` (maps the "text" field to "content" for embeddings)
- **Cloud Provider**: AWS (us-east-1)

**CLI Command:**
```bash
pc index create -n <index_name> -m cosine -c aws -r us-east-1 \
  --model llama-text-embed-v2 --field_map text=content
```

### Data Constraints

- **Flat Metadata Only**: Nested objects are not supported in metadata (only str, int, float, bool)
- **Content Truncation**: Content is truncated to 1000 characters for embedding
- **Text Field Requirement**: Records must include a "text" field that matches the field_map configuration

### Search Optimization

- **Reranking**: Uses bge-reranker-v2-m3 for better result quality
- **Candidate Expansion**: Fetches 2x the requested results before reranking
- **Wait After Upsert**: Search results may not be immediately available after upserting (10+ second delay recommended)

### Global Instance

The module exports a global instance:
```python
memory_store = PineconeStore()
```

This allows easy importing across the codebase:
```python
from memory.pinecone_store import memory_store
```

---

## Usage Examples

### Storing Code
```python
from memory.pinecone_store import memory_store

memory_store.store_code(
    code="def hello(): print('world')",
    file_path="src/main.py",
    commit_hash="abc123",
    metadata={"language": "python"}
)
```

### Storing a Decision
```python
memory_store.store_decision(
    decision="Use FastAPI for the API layer",
    context="Need REST API with async support",
    agent="architect-agent",
    metadata={"priority": "high"}
)
```

### Searching for Similar Code
```python
results = memory_store.search_similar(
    query="authentication middleware",
    top_k=5,
    filter_dict={"type": "code"}
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"File: {result['metadata']['file_path']}")
```

### Getting Context for LLM
```python
context = memory_store.get_relevant_context(
    query="How to handle database errors?",
    max_results=5
)
# Use context in LLM prompt
```

### Storing Error Patterns
```python
memory_store.store_error_pattern(
    error="ConnectionError: database connection timeout",
    fix="Increase connection pool size and add retry logic",
    metadata={"component": "database", "severity": "high"}
)
```

---

## Important Notes

1. **Index Setup**: The index must be created before using this class (via Pinecone CLI)
2. **API Key**: Requires `PINECONE_API_KEY` in environment variables
3. **Namespace Isolation**: Use different namespaces for different projects/environments
4. **Search Delay**: Wait 10+ seconds after upserting before searching for best results
5. **Metadata Flattening**: Only flat metadata is supported (no nested dictionaries)
6. **Content Limits**: Content is automatically truncated to 1000 characters

---

## Integration Points

This module integrates with:
- `config.settings`: For API keys and index configuration
- `agents/lrm_agent.py`: For storing reasoning traces
- `orchestration/graph.py`: For storing plans and decisions
- Various tools: For storing code and error patterns

---

## Future Improvements

Potential enhancements:
- Support for batch operations
- Update/delete methods
- Namespace management utilities
- Content chunking for large files
- Compression for stored content

