"""GitHub API and Git CLI tools."""
import subprocess
from typing import Optional, Dict, List, Any
from github import Github
from config.settings import settings


class GitHubTool:
    """GitHub API and Git CLI operations."""
    
    def __init__(self):
        self.token = settings.github_token
        self.repo_name = settings.github_repo
        self.base_branch = settings.github_base_branch
        self.github = Github(self.token) if self.token else None
        self.repo = None
        
        if self.github and self.repo_name:
            try:
                self.repo = self.github.get_repo(self.repo_name)
            except Exception as e:
                # Repository not found or not accessible - continue without repo connection
                # The tool can still use git CLI commands even without API access
                print(f"Warning: Could not connect to GitHub repository '{self.repo_name}': {e}")
                print("GitHub API features will be disabled, but git CLI commands will still work.")
                self.repo = None
    
    def _run_git_command(self, command: List[str]) -> tuple[str, str, int]:
        """Run git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + command,
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout, result.stderr, result.returncode
        except Exception as e:
            return "", str(e), 1
    
    def pull_latest(self, branch: Optional[str] = None) -> tuple[bool, str]:
        """Pull latest changes from remote."""
        branch = branch or self.base_branch
        stdout, stderr, code = self._run_git_command(["pull", "origin", branch])
        return code == 0, stdout + stderr
    
    def create_branch(self, branch_name: str) -> bool:
        """Create and checkout a new branch."""
        _, _, code = self._run_git_command(["checkout", "-b", branch_name])
        return code == 0
    
    def get_current_branch(self) -> str:
        """Get current branch name."""
        stdout, _, code = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        return stdout.strip() if code == 0 else ""
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> bool:
        """Commit changes with message."""
        # Stage files
        if files:
            for file in files:
                self._run_git_command(["add", file])
        else:
            self._run_git_command(["add", "."])
        
        # Commit
        _, _, code = self._run_git_command(
            ["commit", "-m", message]
        )
        return code == 0
    
    def push_branch(self, branch_name: Optional[str] = None) -> bool:
        """Push branch to remote."""
        branch = branch_name or self.get_current_branch()
        _, _, code = self._run_git_command(
            ["push", "-u", "origin", branch]
        )
        return code == 0
    
    def get_diff(self, base_branch: Optional[str] = None) -> str:
        """Get diff between current branch and base."""
        base = base_branch or self.base_branch
        stdout, _, code = self._run_git_command(
            ["diff", f"origin/{base}"]
        )
        return stdout if code == 0 else ""
    
    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a pull request via GitHub API."""
        if not self.repo:
            return None
        
        base = base_branch or self.base_branch
        
        try:
            pr = self.repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base
            )
            return {
                "number": pr.number,
                "url": pr.html_url,
                "state": pr.state,
                "title": pr.title
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_pr_status(self, pr_number: int) -> Dict[str, Any]:
        """Get PR status including CI checks."""
        if not self.repo:
            return {"error": "Repository not initialized"}
        
        try:
            pr = self.repo.get_pull(pr_number)
            checks = pr.get_check_runs()
            
            status = {
                "number": pr.number,
                "state": pr.state,
                "mergeable": pr.mergeable,
                "checks": []
            }
            
            for check in checks:
                status["checks"].append({
                    "name": check.name,
                    "status": check.status,
                    "conclusion": check.conclusion,
                    "url": check.html_url
                })
            
            return status
        except Exception as e:
            return {"error": str(e)}
    
    def get_pr_comments(self, pr_number: int) -> List[Dict[str, Any]]:
        """Get comments on a PR."""
        if not self.repo:
            return []
        
        try:
            pr = self.repo.get_pull(pr_number)
            comments = []
            for comment in pr.get_issue_comments():
                comments.append({
                    "body": comment.body,
                    "user": comment.user.login,
                    "created_at": comment.created_at.isoformat()
                })
            return comments
        except Exception:
            return []
    
    def get_file_contents(self, file_path: str, ref: Optional[str] = None) -> Optional[str]:
        """Get file contents from repository."""
        if not self.repo:
            return None
        
        try:
            ref = ref or self.base_branch
            contents = self.repo.get_contents(file_path, ref=ref)
            if contents.encoding == "base64":
                import base64
                return base64.b64decode(contents.content).decode("utf-8")
            return contents.content
        except Exception:
            return None
    
    def list_files(self, path: str = "", ref: Optional[str] = None) -> List[str]:
        """List files in repository."""
        if not self.repo:
            return []
        
        try:
            ref = ref or self.base_branch
            contents = self.repo.get_contents(path, ref=ref)
            
            if isinstance(contents, list):
                return [item.path for item in contents]
            else:
                return [contents.path]
        except Exception:
            return []

