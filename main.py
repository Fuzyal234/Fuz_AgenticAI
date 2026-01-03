"""Main entry point for FUZ_AgenticAI."""
import sys
import argparse
from orchestration.graph import AgentOrchestrator
from config.settings import settings


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
    if args.request:
        user_request = args.request
    else:
        print("🤖 FUZ_AgenticAI - Autonomous Code Modification & PR Agent this is the testing flow working")
        print("\nEnter your request (or Ctrl+D to exit):")
        try:
            user_request = input("> ")
        except EOFError:
            print("\nExiting...")
            sys.exit(0)
    
    if not user_request.strip():
        print("Error: Request cannot be empty")
        sys.exit(1)
    
    # Validate configuration (only if not using Ollama)
    if not settings.use_ollama:
        if not settings.openai_api_key:
            print("Error: Either USE_OLLAMA=true or OPENAI_API_KEY must be set")
            print("  - For free local LLM: Set USE_OLLAMA=true in .env")
            print("  - For OpenAI/OpenRouter: Set OPENAI_API_KEY in .env")
            sys.exit(1)
    
    if not settings.pinecone_api_key:
        print("Warning: PINECONE_API_KEY not set - memory features will be limited")
    
    if not settings.github_token:
        print("Warning: GITHUB_TOKEN not set - GitHub operations will fail")
    
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
        
        # Print results
        print("\n" + "="*60)
        print("📋 EXECUTION SUMMARY")
        print("="*60)
        
        if result.get("pr_url"):
            print(f"✅ Pull Request Created: {result['pr_url']}")
        
        # Set default status if not set
        final_status = result.get('final_status')
        if not final_status:
            # Determine status based on what happened
            if result.get("pr_url"):
                final_status = "pr_created"
            elif result.get("code_changes"):
                final_status = "code_changes_made"
            elif result.get("plan"):
                final_status = "planning_completed"
            else:
                final_status = "completed"
            result['final_status'] = final_status
        
        if final_status == "success":
            print("✅ Status: SUCCESS")
        elif final_status == "ci_failed":
            print("⚠️  Status: CI FAILED (check PR for details)")
        elif final_status == "pr_created":
            print("✅ Status: PR CREATED")
        elif final_status == "code_changes_made":
            print("✅ Status: CODE CHANGES MADE")
        elif final_status == "planning_completed":
            print("ℹ️  Status: PLANNING COMPLETED (no execution needed)")
        else:
            print(f"ℹ️  Status: {final_status.upper()}")
        
        if result.get("errors"):
            print(f"\n⚠️  Errors encountered: {len(result['errors'])}")
            for error in result["errors"][:5]:  # Show first 5 errors
                print(f"   - {error[:100]}")
        
        if result.get("code_changes"):
            print(f"\n📝 Files modified: {len(result['code_changes'])}")
            for change in result["code_changes"]:
                print(f"   - {change.get('file_path')}")
        
        print(f"\n🔄 Iterations: {result.get('iterations', 0)}")
        print("="*60)
        
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

