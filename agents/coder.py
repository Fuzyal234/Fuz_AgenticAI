"""Code Agent - Modifies and creates code files."""
from typing import Dict, List, Any, Optional
import os
from config.llm import llm_client
from memory.pinecone_store import memory_store
from tools.github_tool import GitHubTool


class CodeAgent:
    """Code agent that modifies and creates code files."""
    
    def __init__(self):
        self.llm = llm_client
        self.memory = memory_store
        self.github_tool = GitHubTool()
    
    def generate_code(
        self,
        task: str,
        file_path: Optional[str] = None,
        existing_code: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate or modify code based on task."""
        
        # Get relevant code examples from memory
        memory_context = self.memory.get_relevant_context(task, max_results=3)
        
        if file_path and existing_code:
            # Modify existing code
            prompt = f"""You are an expert software engineer. Modify the following code to accomplish the task.

Task: {task}

File: {file_path}

Current Code:
{existing_code}

Repository Context:
{context or "No additional context"}

Relevant Examples:
{memory_context}

Instructions:
1. Make minimal, focused changes
2. Follow the existing code style and patterns
3. Add comments where necessary
4. Ensure the code is production-ready

Return ONLY the modified code, no explanations or markdown formatting.
"""
        else:
            # Create new code
            prompt = f"""You are an expert software engineer. Create code to accomplish the task.

Task: {task}

File: {file_path or "new file"}

Repository Context:
{context or "No additional context"}

Relevant Examples:
{memory_context}

Instructions:
1. Write clean, production-ready code
2. Follow best practices
3. Add appropriate comments and docstrings
4. Include error handling where necessary

Return ONLY the code, no explanations or markdown formatting.
"""
        
        messages = [
            {"role": "system", "content": "You are an expert software engineer. Return only code, no markdown or explanations."},
            {"role": "user", "content": prompt}
        ]
        
        generated_code = self.llm.chat_completion(messages, temperature=0.2)
        
        # Clean up the response (remove markdown code blocks if present)
        if "```" in generated_code:
            lines = generated_code.split("\n")
            code_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or not line.strip().startswith("```"):
                    code_lines.append(line)
            generated_code = "\n".join(code_lines)
        
        return {
            "code": generated_code.strip(),
            "file_path": file_path,
            "task": task
        }
    
    def apply_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply code changes to files."""
        results = []
        
        for change in changes:
            file_path = change.get("file_path")
            code = change.get("code")
            
            if not file_path or not code:
                results.append({
                    "file_path": file_path,
                    "success": False,
                    "error": "Missing file_path or code"
                })
                continue
            
            try:
                # Create directory if needed
                os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
                
                # Write code to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Store in memory
                self.memory.store_code(
                    code=code,
                    file_path=file_path,
                    metadata={"agent": "coder", "task": change.get("task", "")}
                )
                
                results.append({
                    "file_path": file_path,
                    "success": True,
                    "message": "File written successfully"
                })
            except Exception as e:
                results.append({
                    "file_path": file_path,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "changes_applied": results,
            "total_files": len(changes),
            "successful": sum(1 for r in results if r.get("success"))
        }
    
    def fix_code(
        self,
        error_message: str,
        file_path: str,
        existing_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fix code based on error message."""
        
        # Check memory for similar error patterns
        error_context = self.memory.search_similar(
            f"Error: {error_message}",
            top_k=3,
            filter_dict={"type": "error_pattern"}
        )
        
        similar_fixes = "\n".join([
            f"Error: {r['metadata'].get('error')}\nFix: {r['metadata'].get('fix')}"
            for r in error_context
        ])
        
        if not existing_code:
            existing_code = self.github_tool.get_file_contents(file_path)
        
        prompt = f"""Fix the following code based on the error message.

Error:
{error_message}

File: {file_path}

Current Code:
{existing_code}

Similar Past Fixes:
{similar_fixes}

Return ONLY the fixed code, no explanations.
"""
        
        messages = [
            {"role": "system", "content": "You are an expert debugger. Return only fixed code."},
            {"role": "user", "content": prompt}
        ]
        
        fixed_code = self.llm.chat_completion(messages, temperature=0.1)
        
        # Clean up response
        if "```" in fixed_code:
            lines = fixed_code.split("\n")
            code_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or not line.strip().startswith("```"):
                    code_lines.append(line)
            fixed_code = "\n".join(code_lines)
        
        # Store error pattern in memory
        self.memory.store_error_pattern(
            error=error_message,
            fix=fixed_code[:500],
            metadata={"file_path": file_path}
        )
        
        return {
            "code": fixed_code.strip(),
            "file_path": file_path
        }

