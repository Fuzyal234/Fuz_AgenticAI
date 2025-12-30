"""Review Agent - Reviews code for quality, bugs, and style."""
from typing import Dict, List, Any, Optional
from config.llm import llm_client
from memory.pinecone_store import memory_store

# Constants for JSON parsing
JSON_CODE_BLOCK_START = "```json"
CODE_BLOCK_START = "```"
CODE_BLOCK_START_LEN_JSON = 7
CODE_BLOCK_START_LEN = 3


class ReviewAgent:
    """Review agent that reviews code for quality and issues."""
    
    def __init__(self):
        self.llm = llm_client
        self.memory = memory_store
    
    def review_code(
        self,
        code: str,
        file_path: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Review code and provide feedback."""
        
        # Get similar code patterns from memory
        memory_context = self.memory.get_relevant_context(code[:500], max_results=2)
        
        prompt = f"""You are a senior code reviewer. Review the following code for:
1. Bugs and logic errors
2. Security issues
3. Code style and conventions
4. Performance issues
5. Best practices

File: {file_path}

Code:
{code}

Repository Context:
{context or "No additional context"}

Similar Code Patterns:
{memory_context}

Provide a comprehensive review in the following JSON format:
{{
    "approved": true|false,
    "issues": [
        {{
            "severity": "critical|high|medium|low",
            "type": "bug|security|style|performance|best_practice",
            "description": "Description of the issue",
            "line": line_number (if applicable),
            "suggestion": "How to fix"
        }}
    ],
    "overall_quality": "excellent|good|needs_improvement|poor",
    "summary": "Overall review summary"
}}
"""
        
        messages = [
            {"role": "system", "content": "You are an expert code reviewer. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_completion(messages, temperature=0.3)
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from response
            if JSON_CODE_BLOCK_START in response:
                json_start = response.find(JSON_CODE_BLOCK_START) + CODE_BLOCK_START_LEN_JSON
                json_end = response.find(CODE_BLOCK_START, json_start)
                response = response[json_start:json_end].strip()
            elif CODE_BLOCK_START in response:
                json_start = response.find(CODE_BLOCK_START) + CODE_BLOCK_START_LEN
                json_end = response.find(CODE_BLOCK_START, json_start)
                response = response[json_start:json_end].strip()
            
            review = json.loads(response)
            
            # Store review decision in memory
            self.memory.store_decision(
                decision=f"Code review for {file_path}: {'Approved' if review.get('approved') else 'Needs fixes'}",
                context=str(review),
                agent="reviewer"
            )
            
            return review
        except json.JSONDecodeError:
            # Fallback review
            return {
                "approved": False,
                "issues": [
                    {
                        "severity": "medium",
                        "type": "review_error",
                        "description": "Unable to parse review response",
                        "suggestion": "Manual review required"
                    }
                ],
                "overall_quality": "needs_improvement",
                "summary": "Review parsing failed, manual review recommended"
            }
    
    def review_changes(
        self,
        diff: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Review a git diff."""
        
        prompt = f"""Review the following code changes (git diff):

{diff}

Context:
{context or "No additional context"}

Provide a review focusing on:
1. Are the changes correct and complete?
2. Do they follow repository conventions?
3. Are there any breaking changes?
4. Is the code style consistent?

Respond in JSON format:
{{
    "approved": true|false,
    "issues": [
        {{
            "severity": "critical|high|medium|low",
            "description": "Issue description",
            "suggestion": "How to fix"
        }}
    ],
    "summary": "Review summary"
}}
"""
        
        messages = [
            {"role": "system", "content": "You are an expert code reviewer. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_completion(messages, temperature=0.3)
        
        # Parse JSON response
        import json
        try:
            if JSON_CODE_BLOCK_START in response:
                json_start = response.find(JSON_CODE_BLOCK_START) + CODE_BLOCK_START_LEN_JSON
                json_end = response.find(CODE_BLOCK_START, json_start)
                response = response[json_start:json_end].strip()
            elif CODE_BLOCK_START in response:
                json_start = response.find(CODE_BLOCK_START) + CODE_BLOCK_START_LEN
                json_end = response.find(CODE_BLOCK_START, json_start)
                response = response[json_start:json_end].strip()
            
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "approved": False,
                "issues": [{"severity": "medium", "description": "Review parsing failed", "suggestion": "Manual review"}],
                "summary": "Unable to parse review"
            }
    
    def should_approve(self, review: Dict[str, Any]) -> bool:
        """Determine if code should be approved based on review."""
        if not review.get("approved", False):
            return False
        
        # Check for critical issues
        issues = review.get("issues", [])
        critical_issues = [
            issue for issue in issues
            if issue.get("severity") == "critical"
        ]
        
        if critical_issues:
            return False
        
        return True

