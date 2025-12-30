"""Test Agent - Runs tests and collects results."""
from typing import Dict, List, Any, Optional
from tools.shell_tool import ShellTool
from tools.ci_tool import CITool


class TestAgent:
    """Test agent that runs tests and collects results."""
    
    def __init__(self):
        self.shell_tool = ShellTool()
        self.ci_tool = CITool()
    
    def run_tests(
        self,
        test_command: str = "pytest",
        cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run tests and return results."""
        success, output = self.shell_tool.run_tests(test_command)
        
        return {
            "success": success,
            "output": output,
            "test_command": test_command,
            "passed": success,
            "failed": not success
        }
    
    def run_linter(
        self,
        linter_command: str = "flake8 .",
        cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run linter and return results."""
        success, output = self.shell_tool.run_linter(linter_command)
        
        return {
            "success": success,
            "output": output,
            "linter_command": linter_command,
            "issues_found": not success
        }
    
    def run_build(
        self,
        build_command: str = "make build",
        cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run build and return results."""
        success, output = self.shell_tool.run_build(build_command)
        
        return {
            "success": success,
            "output": output,
            "build_command": build_command
        }
    
    def check_ci_status(self, pr_number: int) -> Dict[str, Any]:
        """Check CI/CD status for a PR."""
        return self.ci_tool.get_ci_status(pr_number)
    
    def extract_test_failures(self, test_output: str) -> List[Dict[str, str]]:
        """Extract test failure information from output."""
        failures = []
        lines = test_output.split("\n")
        
        current_failure = None
        for i, line in enumerate(lines):
            if "FAILED" in line or "ERROR" in line:
                if current_failure:
                    failures.append(current_failure)
                
                # Try to extract test name
                test_name = line.split("::")[-1] if "::" in line else line.strip()
                current_failure = {
                    "test_name": test_name,
                    "error_message": "",
                    "line_number": i + 1
                }
            elif current_failure and line.strip():
                current_failure["error_message"] += line + "\n"
        
        if current_failure:
            failures.append(current_failure)
        
        return failures
    
    def get_failure_summary(self, test_results: Dict[str, Any]) -> str:
        """Get a summary of test failures."""
        if test_results.get("success"):
            return "All tests passed"
        
        failures = self.extract_test_failures(test_results.get("output", ""))
        
        if not failures:
            return f"Tests failed but no specific failures extracted:\n{test_results.get('output', '')[:500]}"
        
        summary = f"Found {len(failures)} test failure(s):\n\n"
        for failure in failures:
            summary += f"Test: {failure.get('test_name')}\n"
            summary += f"Error: {failure.get('error_message', '')[:200]}\n\n"
        
        return summary

