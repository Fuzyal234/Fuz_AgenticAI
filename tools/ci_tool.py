"""CI/CD status and feedback tools."""
from typing import Dict, List, Optional, Any
from tools.github_tool import GitHubTool


class CITool:
    """CI/CD pipeline status and feedback."""
    
    def __init__(self):
        self.github_tool = GitHubTool()
    
    def get_ci_status(self, pr_number: int) -> Dict[str, Any]:
        """Get CI/CD status for a PR."""
        pr_status = self.github_tool.get_pr_status(pr_number)
        
        if "error" in pr_status:
            return pr_status
        
        checks = pr_status.get("checks", [])
        
        # Determine overall status
        all_passed = all(
            check.get("conclusion") == "success"
            for check in checks
        )
        
        failed_checks = [
            check for check in checks
            if check.get("conclusion") in ["failure", "error"]
        ]
        
        return {
            "overall_status": "success" if all_passed else "failure",
            "all_passed": all_passed,
            "total_checks": len(checks),
            "failed_checks": failed_checks,
            "check_details": checks
        }
    
    def get_ci_logs(self, pr_number: int, check_name: Optional[str] = None) -> str:
        """Get CI logs for failed checks."""
        ci_status = self.get_ci_status(pr_number)
        
        if ci_status.get("overall_status") == "success":
            return "All CI checks passed"
        
        failed_checks = ci_status.get("failed_checks", [])
        
        if check_name:
            failed_checks = [
                check for check in failed_checks
                if check.get("name") == check_name
            ]
        
        logs = []
        for check in failed_checks:
            logs.append(
                f"Check: {check.get('name')}\n"
                f"Status: {check.get('conclusion')}\n"
                f"URL: {check.get('url')}\n"
            )
        
        return "\n".join(logs) if logs else "No failed checks found"
    
    def wait_for_ci(self, pr_number: int, max_wait_time: int = 600) -> Dict[str, Any]:
        """Wait for CI to complete (simplified - in production use polling)."""
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.get_ci_status(pr_number)
            
            # Check if all checks are complete
            checks = status.get("check_details", [])
            if checks:
                all_complete = all(
                    check.get("status") == "completed"
                    for check in checks
                )
                
                if all_complete:
                    return status
            
            time.sleep(10)  # Poll every 10 seconds
        
        return {
            "overall_status": "timeout",
            "message": "CI checks did not complete within timeout"
        }
    
    def extract_errors_from_logs(self, logs: str) -> List[Dict[str, str]]:
        """Extract error patterns from CI logs."""
        errors = []
        lines = logs.split("\n")
        
        current_error = None
        for line in lines:
            if any(keyword in line.lower() for keyword in ["error:", "failed:", "exception:"]):
                if current_error:
                    errors.append(current_error)
                current_error = {"type": "error", "message": line}
            elif current_error and line.strip():
                current_error["message"] += "\n" + line
        
        if current_error:
            errors.append(current_error)
        
        return errors

