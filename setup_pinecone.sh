#!/bin/bash
# Pinecone Setup Script for FUZ_AgenticAI

set -e

echo "🚀 Setting up Pinecone for FUZ_AgenticAI..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
    echo ""
fi

# Check if Pinecone CLI is installed
if ! command -v pc &> /dev/null; then
    echo "📦 Installing Pinecone CLI..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew tap pinecone-io/tap
        brew install pinecone-io/tap/pinecone
    else
        echo "⚠️  Please install Pinecone CLI manually from: https://github.com/pinecone-io/cli/releases"
        exit 1
    fi
else
    echo "✅ Pinecone CLI already installed"
fi

# Verify CLI installation
echo ""
echo "🔍 Verifying Pinecone CLI..."
pc version

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo ""
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo "   Activate it with: source venv/bin/activate"
else
    echo "✅ Virtual environment already exists"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Dependencies installed"

# Check if API key is set
echo ""
if [ -f .env ]; then
    source .env
    if [ -z "$PINECONE_API_KEY" ] || [ "$PINECONE_API_KEY" == "your_pinecone_api_key_here" ]; then
        echo "⚠️  WARNING: PINECONE_API_KEY not set in .env file"
        echo "   Get your API key from: https://app.pinecone.io/"
    else
        echo "✅ PINECONE_API_KEY is set"
        
        # Configure CLI auth
        echo ""
        echo "🔐 Configuring Pinecone CLI authentication..."
        export PINECONE_API_KEY
        pc auth configure --api-key "$PINECONE_API_KEY" || echo "CLI auth already configured"
    fi
else
    echo "⚠️  .env file not found. Please create it from .env.example"
fi

# Create Pinecone index
echo ""
echo "📊 Creating Pinecone index..."
INDEX_NAME=$(grep PINECONE_INDEX_NAME .env 2>/dev/null | cut -d '=' -f2 || echo "fuz-agentic-ai")

if [ -n "$PINECONE_API_KEY" ] && [ "$PINECONE_API_KEY" != "your_pinecone_api_key_here" ]; then
    export PINECONE_API_KEY
    echo "Creating index: $INDEX_NAME"
    pc index create -n "$INDEX_NAME" -m cosine -c aws -r us-east-1 \
        --model llama-text-embed-v2 \
        --field_map text=content || echo "Index may already exist (this is OK)"
    
    echo ""
    echo "⏳ Waiting for index to be ready..."
    sleep 5
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Make sure your .env file has all required API keys"
    echo "   2. Activate virtual environment: source venv/bin/activate"
    echo "   3. Test the setup: python -c 'from memory.pinecone_store import memory_store; print(\"✅ Pinecone connected!\")'"
else
    echo ""
    echo "⚠️  Please set PINECONE_API_KEY in .env file first"
    echo "   Then run this script again or create the index manually:"
    echo "   pc index create -n $INDEX_NAME -m cosine -c aws -r us-east-1 --model llama-text-embed-v2 --field_map text=content"
fi

