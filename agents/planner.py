"""Planner Agent - Understands requests and creates execution plans."""
from typing import Dict, List, Any, Optional
from config.llm import llm_client
from memory.pinecone_store import memory_store


class PlannerAgent:
    """Planner agent that breaks down user requests into actionable steps."""
    
    def __init__(self):
        self.llm = llm_client
        self.memory = memory_store
    
    def plan(
        self,
        user_request: str,
        repo_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an execution plan for the user request."""
        
        # Get relevant context from memory
        memory_context = self.memory.get_relevant_context(user_request, max_results=5)
        
        planning_prompt = f"""You are a software engineering planner. Your task is to break down a user request into actionable steps.

User Request: {user_request}

Repository Context:
{repo_context or "No context provided"}

Relevant Past Decisions/Code:
{memory_context}

Create a detailed execution plan with the following structure:
1. Understand the request and its requirements
2. Identify which files need to be modified or created
3. Determine what tests need to be run
4. Plan the implementation steps
5. Identify potential risks or issues

Respond in the following JSON format:
{{
    "understanding": "Brief understanding of the request",
    "steps": [
        {{
            "agent": "coder|tester|reviewer",
            "action": "Description of what this agent should do",
            "files": ["list of files to work on"],
            "dependencies": ["any dependencies on previous steps"]
        }}
    ],
    "estimated_complexity": "low|medium|high",
    "risks": ["list of potential risks"]
}}
"""
        
        messages = [
            {"role": "system", "content": "You are an expert software engineering planner. Always respond with valid JSON."},
            {"role": "user", "content": planning_prompt}
        ]
        
        response = self.llm.chat_completion(messages, temperature=0.3)
        
        # Store the planning decision in memory
        self.memory.store_decision(
            decision=f"Plan for: {user_request}",
            context=response,
            agent="planner"
        )
        
        # Parse JSON response (simplified - in production use proper JSON parsing)
        import json
        try:
            # Extract JSON from response if it's wrapped in markdown
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            plan = json.loads(response)
            return plan
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "understanding": user_request,
                "steps": [
                    {
                        "agent": "coder",
                        "action": user_request,
                        "files": [],
                        "dependencies": []
                    }
                ],
                "estimated_complexity": "medium",
                "risks": ["Unable to parse plan, proceeding with basic execution"]
            }
    
    def should_continue(self, current_state: Dict[str, Any]) -> bool:
        """Determine if execution should continue based on current state."""
        # Check if max iterations reached
        iterations = current_state.get("iterations", 0)
        max_iterations = current_state.get("max_iterations", 10)
        
        if iterations >= max_iterations:
            return False
        
        # Check if all steps are complete
        steps = current_state.get("plan", {}).get("steps", [])
        completed_steps = current_state.get("completed_steps", [])
        
        if len(completed_steps) >= len(steps):
            return False
        
        # Check if there are blocking errors
        errors = current_state.get("errors", [])
        if errors and not current_state.get("enable_auto_fix", True):
            return False
        
        return True

