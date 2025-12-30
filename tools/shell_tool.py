"""Shell command execution tool with safety checks."""
import subprocess
import shlex
from typing import List, Tuple, Optional
from config.settings import settings


class ShellTool:
    """Safe shell command execution with allow-list."""
    
    def __init__(self):
        self.allowed_commands = settings.allowed_commands
    
    def _is_command_allowed(self, command: str) -> bool:
        """Check if command is in allow-list."""
        parts = shlex.split(command)
        if not parts:
            return False
        
        base_command = parts[0]
        return base_command in self.allowed_commands
    
    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Tuple[str, str, int]:
        """Execute shell command safely."""
        if not self._is_command_allowed(command):
            return (
                "",
                f"Command '{command.split()[0]}' not in allow-list",
                1
            )
        
        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                check=False
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1
    
    def run_tests(self, test_command: str = "pytest") -> Tuple[bool, str]:
        """Run tests and return success status and output."""
        stdout, stderr, code = self.execute(test_command)
        success = code == 0
        output = stdout + stderr
        return success, output
    
    def run_linter(self, linter_command: str = "flake8 .") -> Tuple[bool, str]:
        """Run linter and return success status and output."""
        stdout, stderr, code = self.execute(linter_command)
        success = code == 0
        output = stdout + stderr
        return success, output
    
    def run_build(self, build_command: str = "make build") -> Tuple[bool, str]:
        """Run build command and return success status and output."""
        stdout, stderr, code = self.execute(build_command)
        success = code == 0
        output = stdout + stderr
        return success, output

