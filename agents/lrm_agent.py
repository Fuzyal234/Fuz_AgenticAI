"""Large Reasoning Model (LRM) Agent - Advanced reasoning for complex tasks."""
from typing import Dict, Any, Optional
from config.llm import lrm_client
from memory.pinecone_store import memory_store


class LRMAgent:
    
    def __init__(self):
        self.llm = lrm_client  # Uses specialized LRM model
        self.memory = memory_store
    
    def reason(
        self,
        problem: str,
        context: Optional[str] = None,
        reasoning_type: str = "general"
    ) -> Dict[str, Any]:
        """Perform deep reasoning on a complex problem.
        
        Args:
            problem: The problem or question to reason about
            context: Additional context for reasoning
            reasoning_type: Type of reasoning (general, architectural, debugging, planning)
        
        Returns:
            Dict with reasoning steps, conclusion, and confidence
        """
        
        # Get relevant past reasoning from memory
        past_reasoning = self.memory.search_similar(
            f"Reasoning: {problem}",
            top_k=3,
            filter_dict={"type": "reasoning"}
        )
        
        past_context = "\n".join([
            f"Past Reasoning: {r['metadata'].get('conclusion', '')}\n"
            f"Steps: {r['metadata'].get('reasoning_steps', '')[:200]}"
            for r in past_reasoning
        ])
        
        reasoning_prompt = self._build_reasoning_prompt(
            problem=problem,
            context=context,
            past_context=past_context,
            reasoning_type=reasoning_type
        )
        
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(reasoning_type)
            },
            {
                "role": "user",
                "content": reasoning_prompt
            }
        ]
        
        # Use chain-of-thought reasoning with higher temperature for creativity
        response = self.llm.chat_completion(
            messages=messages,
            temperature=0.4,  # Balanced for reasoning
            max_tokens=4000  # Allow longer reasoning chains
        )
        
        # Parse reasoning response
        reasoning_result = self._parse_reasoning_response(response, reasoning_type)
        
        # Store reasoning trace in memory
        self.memory.store_reasoning_trace(
            problem=problem,
            reasoning_steps=reasoning_result.get("steps", []),
            conclusion=reasoning_result.get("conclusion", ""),
            confidence=reasoning_result.get("confidence", 0.5),
            reasoning_type=reasoning_type
        )
        
        return reasoning_result
    
    def _build_reasoning_prompt(
        self,
        problem: str,
        context: Optional[str],
        past_context: str,
        reasoning_type: str
    ) -> str:
        """Build reasoning prompt based on type."""
        
        base_prompt = f"""Problem: {problem}

Context:
{context or "No additional context provided"}

Past Similar Reasoning:
{past_context or "No similar past reasoning found"}

Please provide a detailed reasoning process following these steps:
1. Break down the problem into sub-problems
2. Analyze each sub-problem systematically
3. Consider multiple approaches/solutions
4. Evaluate trade-offs
5. Reach a conclusion with confidence level

Format your response as JSON:
{{
    "steps": [
        {{
            "step_number": 1,
            "description": "What you're reasoning about in this step",
            "analysis": "Your analysis and thought process",
            "conclusion": "Conclusion from this step"
        }}
    ],
    "overall_conclusion": "Final conclusion or recommendation",
    "confidence": 0.0-1.0,
    "alternative_approaches": ["alternative 1", "alternative 2"],
    "risks": ["potential risk 1", "potential risk 2"]
}}
"""
        
        if reasoning_type == "architectural":
            return f"""{base_prompt}

Focus on:
- System architecture implications
- Scalability considerations
- Maintainability impact
- Technology choices
- Integration points
"""
        elif reasoning_type == "debugging":
            return f"""{base_prompt}

Focus on:
- Root cause analysis
- Error patterns
- System state analysis
- Potential fixes
- Prevention strategies
"""
        elif reasoning_type == "planning":
            return f"""{base_prompt}

Focus on:
- Task decomposition
- Dependency analysis
- Resource requirements
- Timeline estimation
- Risk mitigation
"""
        
        return base_prompt
    
    def _get_system_prompt(self, reasoning_type: str) -> str:
        """Get system prompt based on reasoning type."""
        
        base_prompt = """You are an expert reasoning system. Your task is to perform deep, 
multi-step reasoning on complex problems. Break down problems systematically, 
consider multiple perspectives, and provide well-reasoned conclusions."""
        
        if reasoning_type == "architectural":
            return f"""{base_prompt} You specialize in software architecture and system design. 
Consider scalability, maintainability, and long-term implications."""
        
        elif reasoning_type == "debugging":
            return f"""{base_prompt} You specialize in debugging and root cause analysis. 
Think like a detective, trace through execution paths, and identify the underlying issues."""
        
        elif reasoning_type == "planning":
            return f"""{base_prompt} You specialize in project planning and task decomposition. 
Break down complex tasks into manageable steps with clear dependencies."""
        
        return base_prompt
    
    def _parse_reasoning_response(
        self,
        response: str,
        reasoning_type: str
    ) -> Dict[str, Any]:
        """Parse reasoning response from LLM."""
        import json
        
        try:
            # Extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            
            # Validate structure
            if "steps" not in result:
                result["steps"] = []
            if "overall_conclusion" not in result:
                result["overall_conclusion"] = "Unable to reach conclusion"
            if "confidence" not in result:
                result["confidence"] = 0.5
            
            return result
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Parsing error",
                        "analysis": "Unable to parse reasoning response",
                        "conclusion": response[:500]
                    }
                ],
                "overall_conclusion": "Reasoning completed but response format was invalid",
                "confidence": 0.3,
                "alternative_approaches": [],
                "risks": ["Response parsing failed"]
            }
    
    def complex_planning(
        self,
        user_request: str,
        repo_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use LRM for complex planning tasks.
        
        Returns a structured plan that can be used by the orchestrator.
        """
        problem = f"Plan the step-by-step implementation of: {user_request}"
        
        # Enhanced prompt for planning that requests structured output
        planning_prompt = f"""You are an expert software engineering planner. Break down the following request into detailed, actionable steps.

User Request: {user_request}

Repository Context:
{repo_context or "No context provided"}

Please provide a comprehensive plan with:
1. Clear understanding of the requirements
2. Step-by-step implementation plan
3. Identification of files that need to be created/modified
4. Testing requirements
5. Potential risks and mitigation strategies

Format your response as JSON with this structure:
{{
    "understanding": "Brief understanding of what needs to be done",
    "steps": [
        {{
            "step_number": 1,
            "description": "What needs to be done in this step",
            "analysis": "Detailed analysis of this step",
            "conclusion": "Specific action to take",
            "agent": "coder|tester|reviewer",
            "files": ["list of files"],
            "dependencies": ["any dependencies"]
        }}
    ],
    "overall_conclusion": "Summary of the recommended approach",
    "confidence": 0.0-1.0,
    "risks": ["potential risk 1", "potential risk 2"],
    "estimated_complexity": "low|medium|high"
}}
"""
        
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt("planning")
            },
            {
                "role": "user",
                "content": planning_prompt
            }
        ]
        
        # Use LRM for planning with higher token limit
        response = self.llm.chat_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for more structured planning
            max_tokens=4000
        )
        
        # Parse the response
        reasoning_result = self._parse_planning_response(response)
        
        # Store reasoning trace
        self.memory.store_reasoning_trace(
            problem=problem,
            reasoning_steps=reasoning_result.get("steps", []),
            conclusion=reasoning_result.get("overall_conclusion", ""),
            confidence=reasoning_result.get("confidence", 0.5),
            reasoning_type="planning"
        )
        
        return {
            "reasoning": reasoning_result,
            "recommended_approach": reasoning_result.get("overall_conclusion", ""),
            "steps": reasoning_result.get("steps", []),
            "confidence": reasoning_result.get("confidence", 0.5),
            "understanding": reasoning_result.get("understanding", ""),
            "risks": reasoning_result.get("risks", []),
            "estimated_complexity": reasoning_result.get("estimated_complexity", "medium")
        }
    
    def _parse_planning_response(self, response: str) -> Dict[str, Any]:
        """Parse planning response from LLM."""
        import json
        
        try:
            # Extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            
            # Validate and normalize structure
            if "steps" not in result:
                result["steps"] = []
            
            # Ensure each step has required fields
            for step in result.get("steps", []):
                if "agent" not in step:
                    # Infer agent from description
                    desc = step.get("description", "").lower()
                    if "test" in desc:
                        step["agent"] = "tester"
                    elif "review" in desc:
                        step["agent"] = "reviewer"
                    else:
                        step["agent"] = "coder"
                
                if "files" not in step:
                    step["files"] = []
                
                if "dependencies" not in step:
                    step["dependencies"] = []
            
            if "overall_conclusion" not in result:
                result["overall_conclusion"] = "Plan created"
            
            if "confidence" not in result:
                result["confidence"] = 0.7
            
            if "understanding" not in result:
                result["understanding"] = "Implementation plan"
            
            if "risks" not in result:
                result["risks"] = []
            
            if "estimated_complexity" not in result:
                result["estimated_complexity"] = "medium"
            
            return result
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return {
                "understanding": "Unable to parse planning response",
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Parse planning response",
                        "analysis": "Response format was invalid",
                        "conclusion": response[:500],
                        "agent": "coder",
                        "files": [],
                        "dependencies": []
                    }
                ],
                "overall_conclusion": "Planning completed but response format was invalid",
                "confidence": 0.3,
                "risks": ["Response parsing failed"],
                "estimated_complexity": "medium"
            }
    
    def architectural_decision(
        self,
        question: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use LRM for architectural decisions."""
        reasoning_result = self.reason(
            problem=question,
            context=context,
            reasoning_type="architectural"
        )
        
        return {
            "decision": reasoning_result.get("overall_conclusion", ""),
            "reasoning": reasoning_result.get("steps", []),
            "alternatives": reasoning_result.get("alternative_approaches", []),
            "risks": reasoning_result.get("risks", []),
            "confidence": reasoning_result.get("confidence", 0.5)
        }
    
    def debug_complex_issue(
        self,
        error_description: str,
        error_logs: Optional[str] = None,
        code_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use LRM for complex debugging."""
        problem = f"Debug the following issue: {error_description}"
        context = f"Error Logs:\n{error_logs or 'No logs provided'}\n\nCode Context:\n{code_context or 'No code context'}"
        
        reasoning_result = self.reason(
            problem=problem,
            context=context,
            reasoning_type="debugging"
        )
        
        return {
            "root_cause": reasoning_result.get("overall_conclusion", ""),
            "reasoning_steps": reasoning_result.get("steps", []),
            "recommended_fix": reasoning_result.get("alternative_approaches", [])[0] if reasoning_result.get("alternative_approaches") else "",
            "confidence": reasoning_result.get("confidence", 0.5)
        }

