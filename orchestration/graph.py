"""LangGraph orchestration for multi-agent workflow."""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agents.planner import PlannerAgent
from agents.coder import CodeAgent
from agents.tester import TestAgent
from agents.reviewer import ReviewAgent
from agents.lrm_agent import LRMAgent
from tools.github_tool import GitHubTool
from tools.ci_tool import CITool
from config.settings import settings


class AgentState(TypedDict):
    """State structure for LangGraph."""
    user_request: str
    plan: Optional[Dict[str, Any]]
    current_step: int
    completed_steps: List[Dict[str, Any]]
    code_changes: List[Dict[str, Any]]
    test_results: Optional[Dict[str, Any]]
    review_results: Optional[Dict[str, Any]]
    pr_number: Optional[int]
    pr_url: Optional[str]
    ci_status: Optional[Dict[str, Any]]
    errors: List[str]
    iterations: int
    max_iterations: int
    enable_auto_fix: bool
    branch_name: Optional[str]
    final_status: Optional[str]
    reasoning_results: Optional[Dict[str, Any]]
    use_lrm: bool


class AgentOrchestrator:
    """LangGraph-based orchestrator for multi-agent workflow."""
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.coder = CodeAgent()
        self.tester = TestAgent()
        self.reviewer = ReviewAgent()
        self.lrm_agent = LRMAgent() if settings.enable_lrm else None
        self.github_tool = GitHubTool()
        self.ci_tool = CITool()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_node)
        if self.lrm_agent:
            workflow.add_node("reason", self._reason_node)
        workflow.add_node("code", self._code_node)
        workflow.add_node("test", self._test_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("github", self._github_node)
        workflow.add_node("ci_check", self._ci_check_node)
        workflow.add_node("fix", self._fix_node)
        
        # Define edges
        workflow.set_entry_point("plan")
        
        # Use LRM for complex planning if enabled
        if self.lrm_agent:
            workflow.add_conditional_edges(
                "plan",
                self._should_use_lrm,
                {
                    "reason": "reason",
                    "code": "code"
                }
            )
            workflow.add_edge("reason", "code")
        else:
            workflow.add_edge("plan", "code")
        workflow.add_conditional_edges(
            "code",
            self._code_decision,  # Changed to _code_decision for better control
            {
                "review": "review",
                "test": "test",
                "github": "github",
                "end": END  # Allow code to end directly if done
            }
        )
        workflow.add_edge("review", "test")
        workflow.add_conditional_edges(
            "test",
            self._test_decision,
            {
                "github": "github",
                "fix": "fix",
                "end": END
            }
        )
        workflow.add_edge("github", "ci_check")
        workflow.add_conditional_edges(
            "ci_check",
            self._ci_decision,
            {
                "success": END,
                "fix": "fix",
                "end": END
            }
        )
        # Make fix go to test instead of code to break infinite loop
        # This prevents: fix -> code -> test -> fix infinite cycle
        workflow.add_edge("fix", "test")
        
        return workflow.compile()
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Planning node."""
        user_request = state["user_request"]
        
        # Get repository context
        repo_context = self._get_repo_context()
        
        # Create plan
        plan = self.planner.plan(user_request, repo_context)
        
        state["plan"] = plan
        state["current_step"] = 0
        state["completed_steps"] = []
        state["iterations"] = state.get("iterations", 0) + 1
        state["use_lrm"] = state.get("use_lrm", settings.enable_lrm)
        
        # Store initial plan in Pinecone (will be enhanced by LRM if enabled)
        from memory.pinecone_store import memory_store
        try:
            memory_store.store_plan(
                user_request=user_request,
                plan=plan,
                metadata={
                    "has_lrm_reasoning": False,
                    "source": "planner"
                }
            )
        except Exception as e:
            # Log but don't fail if Pinecone storage fails
            print(f"Warning: Failed to store initial plan in Pinecone: {e}")
        
        return state
    
    def _reason_node(self, state: AgentState) -> AgentState:
        """LRM reasoning node for complex tasks."""
        if not self.lrm_agent:
            return state
        
        user_request = state["user_request"]
        plan = state.get("plan", {})
        
        # Use LRM for complex planning
        repo_context = self._get_repo_context()
        reasoning_result = self.lrm_agent.complex_planning(
            user_request=user_request,
            repo_context=repo_context
        )
        
        # Enhance plan with LRM reasoning
        if reasoning_result.get("confidence", 0) > 0.5:
            # Use LRM's structured plan
            lrm_reasoning = reasoning_result.get("reasoning", {})
            lrm_steps = lrm_reasoning.get("steps", [])
            
            # If LRM provided detailed steps with agent and files, use them
            if lrm_steps and len(lrm_steps) > 0:
                # Check if steps have the required structure (agent, files, etc.)
                has_structure = all(
                    "agent" in step and "files" in step 
                    for step in lrm_steps
                )
                
                if has_structure:
                    # LRM provided fully structured steps - use them directly
                    enhanced_steps = []
                    for lrm_step in lrm_steps:
                        enhanced_steps.append({
                            "agent": lrm_step.get("agent", "coder"),
                            "action": lrm_step.get("description", "") + ": " + lrm_step.get("conclusion", ""),
                            "files": lrm_step.get("files", []),
                            "dependencies": lrm_step.get("dependencies", []),
                            "lrm_analysis": lrm_step.get("analysis", "")
                        })
                    
                    # Replace plan steps with LRM's structured steps if confidence is high
                    if reasoning_result.get("confidence", 0) > 0.7:
                        plan["steps"] = enhanced_steps
                    else:
                        # Merge: prepend LRM steps to existing plan
                        plan["steps"] = enhanced_steps + plan.get("steps", [])
                else:
                    # LRM steps don't have full structure - convert them
                    enhanced_steps = []
                    for lrm_step in lrm_steps:
                        action = lrm_step.get("description", "") + ": " + lrm_step.get("conclusion", "")
                        analysis = lrm_step.get("analysis", "")
                        
                        # Infer agent from description/analysis
                        desc_lower = (lrm_step.get("description", "") + " " + analysis).lower()
                        agent = "coder"  # Default
                        if "test" in desc_lower or "verify" in desc_lower:
                            agent = "tester"
                        elif "review" in desc_lower or "check" in desc_lower:
                            agent = "reviewer"
                        
                        enhanced_steps.append({
                            "agent": agent,
                            "action": action,
                            "files": lrm_step.get("files", []),
                            "dependencies": lrm_step.get("dependencies", []),
                            "lrm_analysis": analysis
                        })
                    
                    # Merge with existing plan
                    if reasoning_result.get("confidence", 0) > 0.7:
                        plan["steps"] = enhanced_steps
                    else:
                        plan["steps"] = enhanced_steps + plan.get("steps", [])
            
            # Update plan metadata with LRM insights
            if reasoning_result.get("understanding"):
                plan["understanding"] = reasoning_result.get("understanding")
            if reasoning_result.get("estimated_complexity"):
                plan["estimated_complexity"] = reasoning_result.get("estimated_complexity")
            if reasoning_result.get("risks"):
                # Merge risks
                existing_risks = plan.get("risks", [])
                plan["risks"] = list(set(existing_risks + reasoning_result.get("risks", [])))
            
            plan["lrm_reasoning"] = lrm_reasoning
            plan["lrm_recommendation"] = reasoning_result.get("recommended_approach", "")
            plan["lrm_confidence"] = reasoning_result.get("confidence", 0.5)
            state["plan"] = plan
        
        state["reasoning_results"] = reasoning_result
        
        # Store the enhanced plan in Pinecone
        from memory.pinecone_store import memory_store
        try:
            memory_store.store_plan(
                user_request=user_request,
                plan=state["plan"],
                metadata={
                    "lrm_confidence": reasoning_result.get("confidence", 0.5),
                    "has_lrm_reasoning": True
                }
            )
        except Exception as e:
            # Log but don't fail if Pinecone storage fails
            print(f"Warning: Failed to store plan in Pinecone: {e}")
        
        return state
    
    def _should_use_lrm(self, state: AgentState) -> str:
        """Decide if LRM should be used for reasoning."""
        # Always use LRM for planning when enabled
        if self.lrm_agent and state.get("use_lrm", True):
            return "reason"
        
        return "code"
    
    def _code_node(self, state: AgentState) -> AgentState:
        """Code generation node."""
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        current_step = state.get("current_step", 0)
        
        # Check if all steps are completed
        if current_step >= len(steps):
            # All steps done - mark as complete to prevent infinite loops
            if not state.get("final_status"):
                state["final_status"] = "steps_completed"
            # Add a flag to prevent re-processing
            state["all_steps_completed"] = True
            return state
        
        # Safety check: prevent infinite loops
        if state.get("iterations", 0) >= state.get("max_iterations", 10):
            state["final_status"] = "max_iterations_reached"
            return state
        
        step = steps[current_step]
        
        if step.get("agent") != "coder":
            state["current_step"] = current_step + 1
            return state
        
        # Get task and files
        task = step.get("action", "")
        files = step.get("files", [])
        
        # Check if this is a Git operation (pull, checkout, branch creation, etc.)
        task_lower = task.lower()
        is_git_operation = any(cmd in task_lower for cmd in ["git pull", "pull", "checkout", "create branch", "branch"])
        
        if is_git_operation and not files:
            # Handle Git-only operations (no code files to modify)
            if "pull" in task_lower:
                # Pull latest changes
                success, output = self.github_tool.pull_latest()
                if not success:
                    state["errors"].append(f"Git pull failed: {output}")
                else:
                    print(f"✅ Pulled latest changes: {output[:100]}")
            
            # Extract branch name from task if creating a branch
            if "branch" in task_lower or "checkout" in task_lower:
                # Try to extract branch name from task
                import re
                branch_match = re.search(r'(?:branch|checkout).*?(\S+[-_]\w+)', task_lower)
                if branch_match:
                    branch_name = branch_match.group(1).upper() if "TESTIMG" in task.upper() else branch_match.group(1)
                else:
                    # Try to find branch name from user request
                    user_request = state.get("user_request", "")
                    branch_match = re.search(r'(\S+[-_]\w+)', user_request.upper())
                    branch_name = branch_match.group(1) if branch_match else f"branch-{current_step}"
                
                success = self.github_tool.create_branch(branch_name)
                if success:
                    state["branch_name"] = branch_name
                    print(f"✅ Created branch: {branch_name}")
                else:
                    state["errors"].append(f"Failed to create branch: {branch_name}")
            
            # Mark step as completed even without code changes
            state["current_step"] = current_step + 1
            state["completed_steps"].append(step)
            return state
        
        # Normal code generation path
        code_changes = []
        
        for file_path in files:
            # Get existing code if file exists
            existing_code = self.github_tool.get_file_contents(file_path)
            
            # Generate code
            change = self.coder.generate_code(
                task=task,
                file_path=file_path,
                existing_code=existing_code,
                context=self._get_repo_context()
            )
            code_changes.append(change)
        
        # Apply changes only if there are code changes
        if code_changes:
            self.coder.apply_changes(code_changes)
            state["code_changes"] = state.get("code_changes", []) + code_changes
        
        state["current_step"] = current_step + 1
        state["completed_steps"].append(step)
        
        return state
    
    def _review_node(self, state: AgentState) -> AgentState:
        """Code review node."""
        code_changes = state.get("code_changes", [])
        
        review_results = []
        
        for change in code_changes:
            review = self.reviewer.review_code(
                code=change.get("code", ""),
                file_path=change.get("file_path", ""),
                context=self._get_repo_context()
            )
            review_results.append({
                "file_path": change.get("file_path"),
                "review": review
            })
        
        state["review_results"] = review_results
        
        # Check if all reviews are approved
        all_approved = all(
            self.reviewer.should_approve(r["review"])
            for r in review_results
        )
        
        if not all_approved:
            state["errors"].append("Code review failed - issues found")
        
        return state
    
    def _test_node(self, state: AgentState) -> AgentState:
        """Test execution node."""
        # Run tests
        test_results = self.tester.run_tests()
        
        state["test_results"] = test_results
        
        if not test_results.get("success"):
            failure_summary = self.tester.get_failure_summary(test_results)
            state["errors"].append(f"Tests failed: {failure_summary}")
        
        return state
    
    def _github_node(self, state: AgentState) -> AgentState:
        """GitHub operations node."""
        # Create branch
        branch_name = f"agentic-ai-{state.get('iterations', 0)}"
        self.github_tool.create_branch(branch_name)
        state["branch_name"] = branch_name
        
        # Commit changes
        commit_message = f"🤖 Agentic AI: {state['user_request'][:50]}"
        files_changed = [c.get("file_path") for c in state.get("code_changes", [])]
        self.github_tool.commit_changes(commit_message, files_changed)
        
        # Push branch
        self.github_tool.push_branch(branch_name)
        
        # Create PR
        pr_body = f"""
## 🤖 Agentic AI Changes

**Request:** {state['user_request']}

**Changes:**
{chr(10).join(f"- {c.get('file_path')}" for c in state.get('code_changes', []))}

**Plan:**
{state.get('plan', {}).get('understanding', 'N/A')}

---
*This PR was automatically created by FUZ_AgenticAI*
"""
        
        pr = self.github_tool.create_pull_request(
            title=f"🤖 {state['user_request'][:60]}",
            body=pr_body,
            head_branch=branch_name
        )
        
        if pr and "number" in pr:
            state["pr_number"] = pr["number"]
            state["pr_url"] = pr.get("url")
        
        return state
    
    def _ci_check_node(self, state: AgentState) -> AgentState:
        """CI/CD status check node."""
        pr_number = state.get("pr_number")
        
        if not pr_number:
            state["errors"].append("No PR number available for CI check")
            return state
        
        # Wait for CI and get status
        ci_status = self.ci_tool.wait_for_ci(pr_number, max_wait_time=300)
        state["ci_status"] = ci_status
        
        if ci_status.get("overall_status") != "success":
            # Get CI logs
            logs = self.ci_tool.get_ci_logs(pr_number)
            state["errors"].append(f"CI failed: {logs[:500]}")
        
        return state
    
    def _fix_node(self, state: AgentState) -> AgentState:
        """Fix code based on errors."""
        # Safety check: stop fixing if max iterations reached
        current_iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 10)
        
        if current_iterations >= max_iterations:
            state["final_status"] = "max_iterations_reached_in_fix"
            state["errors"].append(f"Reached max iterations ({max_iterations}) during fix attempts")
            return state
        
        errors = state.get("errors", [])
        code_changes = state.get("code_changes", [])
        
        if not errors or not code_changes:
            return state
        
        # Use LRM for complex debugging if enabled
        if self.lrm_agent and len(errors) > 1:
            # Multiple errors suggest complex issue - use LRM
            error_description = "\n".join(errors[-3:])
            error_logs = state.get("test_results", {}).get("output", "")
            
            # Get code context
            code_context = "\n".join([
                f"File: {c.get('file_path')}\n{c.get('code', '')[:500]}"
                for c in code_changes[:3]
            ])
            
            debug_result = self.lrm_agent.debug_complex_issue(
                error_description=error_description,
                error_logs=error_logs,
                code_context=code_context
            )
            
            # Use LRM's root cause analysis
            if debug_result.get("confidence", 0) > 0.6:
                # High confidence - use LRM's recommended fix
                root_cause = debug_result.get("root_cause", "")
                recommended_fix = debug_result.get("recommended_fix", "")
                
                # Apply LRM's reasoning to fixes
                for change in code_changes:
                    fixed = self.coder.fix_code(
                        error_message=f"{root_cause}\nRecommended: {recommended_fix}",
                        file_path=change.get("file_path", ""),
                        existing_code=change.get("code", "")
                    )
                    change["code"] = fixed.get("code", change.get("code"))
        else:
            # Standard fixing without LRM
            for error in errors[-3:]:
                for change in code_changes:
                    fixed = self.coder.fix_code(
                        error_message=error,
                        file_path=change.get("file_path", ""),
                        existing_code=change.get("code", "")
                    )
                    change["code"] = fixed.get("code", change.get("code"))
        
        # Re-apply fixes
        self.coder.apply_changes(code_changes)
        
        # Clear errors for retry
        state["errors"] = []
        state["iterations"] = state.get("iterations", 0) + 1
        
        return state
    
    def _code_decision(self, state: AgentState) -> str:
        """Decide where to go after code node - replaces _should_review for better control."""
        # CRITICAL: Check max iterations FIRST to prevent infinite loops
        current_iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 10)
        
        if current_iterations >= max_iterations:
            state["final_status"] = "max_iterations_reached_in_code"
            return "end"  # End directly
        
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        current_step = state.get("current_step", 0)
        
        # If all steps completed AND we have code changes
        if current_step >= len(steps) and state.get("code_changes"):
            # All steps done
            # If we've already tested multiple times, skip to github/end
            if state.get("test_results") and current_iterations >= max_iterations - 1:
                return "github"  # Skip directly to github
            # If we've tested and failed too many times, end
            if state.get("test_results") and current_iterations >= max_iterations:
                return "end"
            # Haven't tested yet, run test once
            if not state.get("test_results"):
                return "test"
            # Already tested once but failed - give it one more try
            return "test"
        
        # If all steps completed but no code changes, end
        if current_step >= len(steps) and not state.get("code_changes"):
            return "end"
        
        if current_step < len(steps):
            step = steps[current_step]
            if step.get("agent") == "reviewer":
                return "review"
        
        # Check if all coding steps are done
        coding_steps = [s for s in steps if s.get("agent") == "coder"]
        completed_coding = len([s for s in state.get("completed_steps", []) if s.get("agent") == "coder"])
        
        if completed_coding >= len(coding_steps) and current_step >= len(steps):
            # All coding done, run test if not already done
            if not state.get("test_results"):
                return "test"
            # Already tested, proceed to github
            return "github"
        
        # More steps to process - route to test which will eventually loop back if needed
        return "test"
    
    def _should_review(self, state: AgentState) -> str:
        """Decide if code should be reviewed."""
        # CRITICAL: Check max iterations FIRST to prevent infinite loops
        current_iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 10)
        
        if current_iterations >= max_iterations:
            state["final_status"] = "max_iterations_reached_in_review"
            # Route directly to github to end workflow
            return "github"
        
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        current_step = state.get("current_step", 0)
        
        # If all steps completed AND we have code changes
        if current_step >= len(steps) and state.get("code_changes"):
            # All steps done
            # If we've already tested multiple times (have test_results and iterations > 0), skip to github
            if state.get("test_results") and state.get("iterations", 0) > 0:
                return "github"  # Skip directly to github to end workflow
            # Haven't tested yet or first test, run test once
            if not state.get("test_results"):
                return "test"
            # Already tested once but failed - if iterations high, go to github
            if state.get("iterations", 0) >= max_iterations - 1:
                return "github"
            return "test"
        
        if current_step < len(steps):
            step = steps[current_step]
            if step.get("agent") == "reviewer":
                return "review"
        
        # Check if all coding steps are done
        coding_steps = [s for s in steps if s.get("agent") == "coder"]
        completed_coding = len([s for s in state.get("completed_steps", []) if s.get("agent") == "coder"])
        
        if completed_coding >= len(coding_steps) and current_step >= len(steps):
            # All coding done, run test if not already done
            if not state.get("test_results"):
                return "test"
            # Already tested, proceed to github
            return "github"
        
        # If there are more steps, continue with code generation
        if current_step < len(steps):
            # More steps to process - need to route back to code
            # But we can't route back to code from here, so route to test
            # This might create a loop, so ensure test_decision handles it
            return "test"
        
        # Default: proceed to test
        return "test"
    
    def _test_decision(self, state: AgentState) -> str:
        """Decide next step after testing."""
        # CRITICAL: Check max iterations FIRST to prevent infinite loops
        current_iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 10)
        
        if current_iterations >= max_iterations:
            state["final_status"] = "max_iterations_reached_in_test"
            return "end"
        
        test_results = state.get("test_results", {})
        
        if test_results.get("success"):
            # Tests passed, proceed to GitHub
            if state.get("pr_number"):
                return "end"  # PR already created
            return "github"
        else:
            # Tests failed - check if we can still fix
            if state.get("enable_auto_fix", True) and current_iterations < max_iterations:
                return "fix"
            # Can't fix anymore - stop
            state["final_status"] = "tests_failed_max_iterations"
            return "end"
    
    def _ci_decision(self, state: AgentState) -> str:
        """Decide next step after CI check."""
        ci_status = state.get("ci_status", {})
        
        if ci_status.get("overall_status") == "success":
            state["final_status"] = "success"
            return "success"
        else:
            # CI failed
            if state.get("enable_auto_fix", True) and state.get("iterations", 0) < state.get("max_iterations", 10):
                return "fix"
            state["final_status"] = "ci_failed"
            return "end"
    
    def _get_repo_context(self) -> str:
        """Get repository context."""
        # List files in repo
        files = self.github_tool.list_files()
        
        # Get contents of key files
        context_parts = []
        for file_path in files[:10]:  # Limit to first 10 files
            content = self.github_tool.get_file_contents(file_path)
            if content:
                context_parts.append(f"File: {file_path}\n{content[:500]}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def run(self, user_request: str, max_iterations: int = 10) -> Dict[str, Any]:
        """Run the orchestrator with a user request."""
        initial_state: AgentState = {
            "user_request": user_request,
            "plan": None,
            "current_step": 0,
            "completed_steps": [],
            "code_changes": [],
            "test_results": None,
            "review_results": None,
            "pr_number": None,
            "pr_url": None,
            "ci_status": None,
            "errors": [],
            "iterations": 0,
            "max_iterations": max_iterations,
            "enable_auto_fix": True,
            "branch_name": None,
            "final_status": None,
            "reasoning_results": None,
            "use_lrm": settings.enable_lrm
        }
        
        # Run the graph with increased recursion limit and iteration tracking
        # 
        # WHAT IS RECURSION IN LANGGRAPH?
        # Recursion = how many times the graph can loop/cycle through nodes before stopping
        # Example: code -> test -> fix -> code -> test -> fix... (looping)
        # If a task keeps failing tests and needs fixing, it creates a loop
        # Default limit is 25 cycles, but complex tasks may need more
        # 
        # SOLUTION: Track iterations in state and force stop when limit reached
        # Set recursion_limit high enough to allow max_iterations retries
        max_recursions = max_iterations * 20  # Allow 20 node transitions per iteration
        config = {"recursion_limit": max_recursions}
        
        try:
            final_state = self.graph.invoke(initial_state, config=config)
        except Exception as e:
            # If recursion limit hit, return current state with error
            if "recursion_limit" in str(e).lower():
                initial_state["final_status"] = "recursion_limit_exceeded"
                initial_state["errors"].append(f"Graph exceeded recursion limit: {max_recursions}")
                return initial_state
            raise
        
        return final_state
        
        return final_state

