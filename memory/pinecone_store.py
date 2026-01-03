"""Pinecone vector store for long-term memory."""
from typing import List, Dict, Optional, Any
import hashlib
import time
from pinecone import Pinecone
from config.settings import settings


class PineconeStore:
    """Pinecone vector store for code embeddings and memory.
    
    Uses the new Pinecone API with integrated embeddings and namespaces.
    All operations require a namespace for data isolation.
    """
    
    # Default namespace for general memory storage
    DEFAULT_NAMESPACE = "agentic-memory"
    
    def __init__(self):
        self.api_key = settings.pinecone_api_key
        self.index_name = settings.pinecone_index_name
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required. Set it in your .env file.")
        
        self.pc = Pinecone(api_key=self.api_key)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index(self):
        """Ensure Pinecone index exists.
        
        Note: Index creation should be done via CLI for production.
        This method checks if index exists and provides guidance if not.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            raise ValueError(
                f"Index '{self.index_name}' does not exist. "
                f"Create it using the Pinecone CLI:\n"
                f"  pc index create -n {self.index_name} -m cosine -c aws -r us-east-1 "
                f"--model llama-text-embed-v2 --field_map text=content"
            )
    
    def _generate_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID for a record."""
        combined = f"{content}{str(metadata)}"
        print(f"Generated ID FOR THE PINECONE STORE FOR FUZAIL: {hashlib.md5(combined.encode()).hexdigest()}")
        return hashlib.md5(combined.encode()).hexdigest()
    
    def store_code(
        self,
        code: str,
        file_path: str,
        commit_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ):
        """Store code in Pinecone using integrated embeddings.
        
        Args:
            code: Code content to store
            file_path: Path to the file
            commit_hash: Optional git commit hash
            metadata: Additional metadata
            namespace: Namespace to store in (defaults to DEFAULT_NAMESPACE)
        """
        namespace = namespace or self.DEFAULT_NAMESPACE
        
        # Prepare record with content field (must match field_map from index creation)
        record = {
            "_id": self._generate_id(code, {"file_path": file_path, **(metadata or {})}),
            "content": code[:1000],
            "text": code[:1000],
        }
        
        # Add metadata fields (flat structure only)
        record["type"] = "code"
        record["file_path"] = file_path
        if commit_hash:
            record["commit_hash"] = commit_hash
        if metadata:
            # Flatten metadata (no nested objects)
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    record[key] = value
        
        # Upsert record (Pinecone will generate embeddings automatically)
        self.index.upsert_records(namespace, [record])
    
    def store_decision(
        self,
        decision: str,
        context: str,
        agent: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ):
        """Store architectural decision or agent decision."""
        namespace = namespace or self.DEFAULT_NAMESPACE
        
        text = f"Decision: {decision}\nContext: {context}"
        
        record = {
            "_id": self._generate_id(text, {"agent": agent, **(metadata or {})}),
            "content": text[:1000],
            "text": text[:1000],
        }
        
        record["type"] = "decision"
        record["agent"] = agent
        record["decision"] = decision[:500]
        record["context"] = context[:500]
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    record[key] = value
        
        self.index.upsert_records(namespace, [record])
    
    def store_error_pattern(
        self,
        error: str,
        fix: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ):
        """Store error pattern and its fix."""
        namespace = namespace or self.DEFAULT_NAMESPACE
        
        text = f"Error: {error}\nFix: {fix}"
        
        record = {
            "_id": self._generate_id(text, metadata or {}),
            "text": text[:1000],  # Must match field_map (text field for embedding)
        }
        
        record["type"] = "error_pattern"
        record["error"] = error[:500]
        record["fix"] = fix[:500]
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    record[key] = value
        
        self.index.upsert_records(namespace, [record])
    
    def store_reasoning_trace(
        self,
        problem: str,
        reasoning_steps: List[Dict[str, Any]],
        conclusion: str,
        confidence: float,
        reasoning_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ):
        """Store reasoning trace from LRM agent.
        
        Args:
            problem: The problem that was reasoned about
            reasoning_steps: List of reasoning steps
            conclusion: Final conclusion
            confidence: Confidence level (0.0-1.0)
            reasoning_type: Type of reasoning (general, architectural, debugging, planning)
            metadata: Additional metadata
            namespace: Namespace to store in
        """
        namespace = namespace or self.DEFAULT_NAMESPACE
        
        # Format reasoning trace as text
        steps_text = "\n".join([
            f"Step {step.get('step_number', i+1)}: {step.get('description', '')}\n"
            f"Analysis: {step.get('analysis', '')}\n"
            f"Conclusion: {step.get('conclusion', '')}"
            for i, step in enumerate(reasoning_steps)
        ])
        
        text = f"Problem: {problem}\nReasoning Steps:\n{steps_text}\nConclusion: {conclusion}"
        
        record = {
            "_id": self._generate_id(text, {"reasoning_type": reasoning_type, **(metadata or {})}),
            "content": text[:1000],
            "text": text[:1000],
        }
        
        record["type"] = "reasoning"
        record["reasoning_type"] = reasoning_type
        record["problem"] = problem[:500]
        record["conclusion"] = conclusion[:500]
        record["confidence"] = confidence
        record["num_steps"] = len(reasoning_steps)
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    record[key] = value
        
        self.index.upsert_records(namespace, [record])
    
    def store_plan(
        self,
        user_request: str,
        plan: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ):
        """Store execution plan in Pinecone.
        
        Args:
            user_request: The original user request
            plan: The execution plan dictionary with steps, understanding, etc.
            metadata: Additional metadata
            namespace: Namespace to store in (defaults to DEFAULT_NAMESPACE)
        """
        namespace = namespace or self.DEFAULT_NAMESPACE
        
        # Format plan as text for embedding
        steps_text = "\n".join([
            f"Step {i+1}: {step.get('action', '')}\n"
            f"Agent: {step.get('agent', 'unknown')}\n"
            f"Files: {', '.join(step.get('files', []))}\n"
            f"Dependencies: {', '.join(step.get('dependencies', []))}"
            for i, step in enumerate(plan.get('steps', []))
        ])
        
        text = f"Request: {user_request}\n\nUnderstanding: {plan.get('understanding', '')}\n\nSteps:\n{steps_text}\n\nComplexity: {plan.get('estimated_complexity', 'medium')}\nRisks: {', '.join(plan.get('risks', []))}"
        
        record = {
            "_id": self._generate_id(text, {"user_request": user_request, **(metadata or {})}),
            "content": text[:1000],
            "text": text[:1000],
        }
        
        record["type"] = "plan"
        record["user_request"] = user_request[:500]
        record["understanding"] = plan.get("understanding", "")[:500]
        record["complexity"] = plan.get("estimated_complexity", "medium")
        record["num_steps"] = len(plan.get("steps", []))
        record["risks"] = ", ".join(plan.get("risks", []))[:500]
        
        # Store each step separately for better searchability
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    record[key] = value
        
        self.index.upsert_records(namespace, [record])
        
        # Also store individual steps for granular search
        for i, step in enumerate(plan.get("steps", [])):
            step_text = f"Step {i+1}: {step.get('action', '')}\nAgent: {step.get('agent', '')}\nFiles: {', '.join(step.get('files', []))}"
            step_record = {
                "_id": self._generate_id(step_text, {"step_number": i+1, "user_request": user_request}),
                "content": step_text[:1000],
                "text": step_text[:1000],
            }
            step_record["type"] = "plan_step"
            step_record["user_request"] = user_request[:500]
            step_record["step_number"] = i + 1
            step_record["agent"] = step.get("agent", "unknown")
            step_record["action"] = step.get("action", "")[:500]
            step_record["files"] = ", ".join(step.get("files", []))[:500]
            
            self.index.upsert_records(namespace, [step_record])
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        wait_after_upsert: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for similar code/decisions using semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            namespace: Namespace to search in (defaults to DEFAULT_NAMESPACE)
            wait_after_upsert: If True, wait 10 seconds (use after upserting)
        
        Returns:
            List of search results with id, score, and metadata
        """
        namespace = namespace or self.DEFAULT_NAMESPACE
        
        # Wait if needed (after upserting, wait 10+ seconds before searching)
        if wait_after_upsert:
            time.sleep(10)
        
        # Build query dict
        query_dict = {
            "top_k": top_k * 2,  # Get more candidates for better results
            "inputs": {
                "text": query  # Field name must match field_map
            }
        }
        
        # Add filter if provided
        if filter_dict:
            query_dict["filter"] = filter_dict
        
        # Perform search with reranking (best practice)
        results = self.index.search(
            namespace=namespace,
            query=query_dict,
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": top_k,
                "rank_fields": ["text"]  # Only one field supported by bge-reranker-v2-m3
            }
        )
        
        # Extract results
        search_results = []
        for hit in results.result.hits:
            search_results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "metadata": {
                    "content": hit.fields.get("text", ""),
                    **{k: v for k, v in hit.fields.items() if k != "text"}
                }
            })
        
        return search_results
    
    def get_relevant_context(self, query: str, max_results: int = 10, namespace: Optional[str] = None) -> str:
        """Get relevant context for a query."""
        results = self.search_similar(query, top_k=max_results, namespace=namespace)
        
        context_parts = []
        for result in results:
            metadata = result["metadata"]
            if metadata.get("type") == "code":
                context_parts.append(
                    f"File: {metadata.get('file_path', 'unknown')}\n"
                    f"Code: {metadata.get('content', '')[:500]}"
            
                )
            elif metadata.get("type") == "decision":
                context_parts.append(
                    f"Decision: {metadata.get('decision', '')}\n"
                    f"Context: {metadata.get('context', '')[:500]}"
                )
            elif metadata.get("type") == "error_pattern":
                context_parts.append(
                    f"Error: {metadata.get('error', '')}\n"
                    f"Fix: {metadata.get('fix', '')}"
                )
            elif metadata.get("type") == "reasoning":
                context_parts.append(
                    f"Problem: {metadata.get('problem', '')}\n"
                    f"Conclusion: {metadata.get('conclusion', '')}\n"
                    f"Confidence: {metadata.get('confidence', 0.5)}"
                )
            elif metadata.get("type") == "plan":
                context_parts.append(
                    f"Plan for: {metadata.get('user_request', '')}\n"
                    f"Understanding: {metadata.get('understanding', '')}\n"
                    f"Steps: {metadata.get('num_steps', 0)}"
                )
            elif metadata.get("type") == "plan_step":
                context_parts.append(
                    f"Step {metadata.get('step_number', 0)}: {metadata.get('action', '')}\n"
                    f"Agent: {metadata.get('agent', '')}\n"
                    f"Files: {metadata.get('files', '')}"
                )
            else:
                # Generic content
                context_parts.append(metadata.get('content', '')[:500])
        
        print(f"CONTEXT PARTS FOR THE PINECONE STORE FOR FUZAIL: {context_parts}")
        print(f"CONTEXT PARTS FOR THE record length STORE FOR FUZAIL: {len(context_parts)}")
        return "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."


# Global memory store instance
memory_store = PineconeStore()

