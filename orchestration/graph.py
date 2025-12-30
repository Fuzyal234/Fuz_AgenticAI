"""LangGraph orchestration for multi-agent workflow."""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agents.planner import PlannerAgent
from agents.coder import CodeAgent
from agents.tester import TestAgent
from agents.reviewer import ReviewAgent
from tools.github_tool import GitHubTool
from tools.ci_tool import CITool


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


class AgentOrchestrator:
    """LangGraph-based orchestrator for multi-agent workflow."""
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.coder = CodeAgent()
        self.tester = TestAgent()
        self.reviewer = ReviewAgent()
        self.github_tool = GitHubTool()
        self.ci_tool = CITool()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("code", self._code_node)
        workflow.add_node("test", self._test_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("github", self._github_node)
        workflow.add_node("ci_check", self._ci_check_node)
        workflow.add_node("fix", self._fix_node)
        
        # Define edges
        workflow.set_entry_point("plan")
        
        workflow.add_edge("plan", "code")
        workflow.add_conditional_edges(
            "code",
            self._should_review,
            {
                "review": "review",
                "test": "test",
                "github": "github"
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
        workflow.add_edge("fix", "code")
        
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
        
        return state
    
    def _code_node(self, state: AgentState) -> AgentState:
        """Code generation node."""
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        current_step = state.get("current_step", 0)
        
        if current_step >= len(steps):
            return state
        
        step = steps[current_step]
        
        if step.get("agent") != "coder":
            state["current_step"] = current_step + 1
            return state
        
        # Get task and files
        task = step.get("action", "")
        files = step.get("files", [])
        
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
        
        # Apply changes
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
        errors = state.get("errors", [])
        code_changes = state.get("code_changes", [])
        
        if not errors or not code_changes:
            return state
        
        # Try to fix based on error messages
        for error in errors[-3:]:  # Focus on last 3 errors
            for change in code_changes:
                fixed = self.coder.fix_code(
                    error_message=error,
                    file_path=change.get("file_path", ""),
                    existing_code=change.get("code", "")
                )
                
                # Update the change
                change["code"] = fixed.get("code", change.get("code"))
        
        # Re-apply fixes
        self.coder.apply_changes(code_changes)
        
        # Clear errors for retry
        state["errors"] = []
        state["iterations"] = state.get("iterations", 0) + 1
        
        return state
    
    def _should_review(self, state: AgentState) -> str:
        """Decide if code should be reviewed."""
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        current_step = state.get("current_step", 0)
        
        if current_step < len(steps):
            step = steps[current_step]
            if step.get("agent") == "reviewer":
                return "review"
        
        # Check if all coding steps are done
        coding_steps = [s for s in steps if s.get("agent") == "coder"]
        completed_coding = len([s for s in state.get("completed_steps", []) if s.get("agent") == "coder"])
        
        if completed_coding >= len(coding_steps):
            return "review"
        
        return "test"
    
    def _test_decision(self, state: AgentState) -> str:
        """Decide next step after testing."""
        test_results = state.get("test_results", {})
        
        if test_results.get("success"):
            # Tests passed, proceed to GitHub
            if state.get("pr_number"):
                return "end"  # PR already created
            return "github"
        else:
            # Tests failed
            if state.get("enable_auto_fix", True) and state.get("iterations", 0) < state.get("max_iterations", 10):
                return "fix"
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
            "final_status": None
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state

