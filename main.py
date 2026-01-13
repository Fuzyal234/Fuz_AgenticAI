"""Main entry point for FUZ_AgenticAI."""
import sys
import argparse
from orchestration.graph import AgentOrchestrator
from config.settings import settings


def _get_user_request(args) -> str:
    """Get user request from args or input."""
    if args.request:
        return args.request
    
    print("🤖 FUZ_AgenticAI - Autonomous Code Modification & PR Agent this is the testing flow working")
    print("\nEnter your request (or Ctrl+D to exit):")
    try:
        return input("> ")
    except EOFError:
        print("\nExiting...")
        sys.exit(0)


def _validate_configuration():
    """Validate required configuration settings."""
    if not settings.use_ollama and not settings.openai_api_key:
        print("Error: Either USE_OLLAMA=true or OPENAI_API_KEY must be set")
        print("  - For free local LLM: Set USE_OLLAMA=true in .env")
        print("  - For OpenAI/OpenRouter: Set OPENAI_API_KEY in .env")
        sys.exit(1)
    
    if not settings.pinecone_api_key:
        print("Warning: PINECONE_API_KEY not set - memory features will be limited")
    
    if not settings.github_token:
        print("Warning: GITHUB_TOKEN not set - GitHub operations will fail")


def _determine_final_status(result: dict) -> str:
    """Determine final status from result."""
    final_status = result.get('final_status')
    if final_status:
        return final_status
    
    if result.get("pr_url"):
        return "pr_created"
    if result.get("code_changes"):
        return "code_changes_made"
    if result.get("plan"):
        return "planning_completed"
    return "completed"


def _print_status(final_status: str):
    """Print status message based on final status."""
    status_messages = {
        "success": "✅ Status: SUCCESS",
        "ci_failed": "⚠️  Status: CI FAILED (check PR for details)",
        "pr_created": "✅ Status: PR CREATED",
        "code_changes_made": "✅ Status: CODE CHANGES MADE",
        "planning_completed": "ℹ️  Status: PLANNING COMPLETED (no execution needed)"
    }
    
    message = status_messages.get(final_status, f"ℹ️  Status: {final_status.upper()}")
    print(message)


def _print_summary(result: dict):
    """Print execution summary."""
    print("\n" + "="*60)
    print("📋 EXECUTION SUMMARY")
    print("="*60)
    
    if result.get("pr_url"):
        print(f"✅ Pull Request Created: {result['pr_url']}")
    
    final_status = _determine_final_status(result)
    result['final_status'] = final_status
    _print_status(final_status)
    
    if result.get("errors"):
        print(f"\n⚠️  Errors encountered: {len(result['errors'])}")
        for error in result["errors"][:5]:
            print(f"   - {error[:100]}")
    
    if result.get("code_changes"):
        print(f"\n📝 Files modified: {len(result['code_changes'])}")
        for change in result["code_changes"]:
            print(f"   - {change.get('file_path')}")
    
    print(f"\n🔄 Iterations: {result.get('iterations', 0)}")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🤖 AgenticAI - Autonomous Code Modification"
    )
    parser.add_argument(
        "request",
        nargs="?",
        help="User request for the agent to fulfill"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.max_iterations,
        help="Maximum number of iterations (default: 10)"
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="Disable automatic fixing on failures"
    )
    
    args = parser.parse_args()
    
    # Get user request
    user_request = _get_user_request(args)
    
    if not user_request.strip():
        print("Error: Request cannot be empty")
        sys.exit(1)
    
    # Validate configuration
    _validate_configuration()
    
    print(f"\n🚀 Processing request: {user_request}")
    print(f"📊 Max iterations: {args.max_iterations}")
    print(f"🔧 Auto-fix: {not args.no_auto_fix}\n")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Run the agent
    try:
        result = orchestrator.run(
            user_request=user_request,
            max_iterations=args.max_iterations
        )
        
        _print_summary(result)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

