#!/usr/bin/env python3
"""Test script for Pinecone integration."""
import sys
import time

def test_pinecone_connection():
    """Test basic Pinecone connection and operations."""
    print("🧪 Testing Pinecone Integration...")
    print("=" * 60)
    
    try:
        from memory.pinecone_store import memory_store
        print("✅ Successfully imported memory_store")
    except ImportError as e:
        print(f"❌ Failed to import memory_store: {e}")
        return False
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Created a .env file with PINECONE_API_KEY")
        print("   2. Created the Pinecone index using the CLI")
        return False
    
    try:
        # Test storing a decision
        print("\n📝 Testing store_decision...")
        memory_store.store_decision(
            decision="Fuzail Ahmed is a good boy and he is a good developer and he is a good engineer",
            context="Fuzail Ahmed is a good table tennis player and he is a good cricket player",
            agent="test_script"
        )
        print("✅ Successfully stored decision")
        
        # Wait for indexing (required for new API)
        print("\n⏳ Waiting 10 seconds for indexing...")
        time.sleep(10)
        
        # Test search
        print("\n🔍 Testing search_similar...")
        results = memory_store.search_similar(
            query="memory storage decision",
            top_k=3
        )
        
        if results:
            print(f"✅ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result['score']:.3f}, Type: {result['metadata'].get('type', 'unknown')}")
        else:
            print("⚠️  No results found (this might be normal if index is empty)")
        
        # Test get_relevant_context
        print("\n📚 Testing get_relevant_context...")
        context = memory_store.get_relevant_context(
            query="test decision",
            max_results=5
        )
        if context and context != "No relevant context found.":
            print(f"✅ Retrieved context ({len(context)} characters)")
        else:
            print("⚠️  No context found")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pinecone_connection()
    sys.exit(0 if success else 1)

