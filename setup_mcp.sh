#!/bin/bash

# Setup script for MCP server integration with RAG LlamaStack
echo "🚀 Setting up MCP server for web content processing..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js (>=16.0.0) first:"
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2)
echo "✅ Node.js version: $NODE_VERSION"

# Install npm dependencies
echo "📦 Installing MCP server dependencies..."
npm install

# Test MCP server installation
echo "🧪 Testing MCP server installation..."
if npx @just-every/mcp-read-website-fast --help &> /dev/null; then
    echo "✅ MCP server installed successfully!"
    
    # Test with a simple URL
    echo "🌐 Testing web content extraction..."
    TEST_URL="https://example.com"
    echo "   Test URL: $TEST_URL"
    
    # Run a quick test (with timeout)
    timeout 10s npx @just-every/mcp-read-website-fast read-website --url "$TEST_URL" --format markdown > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Web content extraction test passed!"
        echo ""
        echo "🎉 MCP server setup complete!"
        echo "   You can now use web URLs in the RAG application"
        echo ""
        echo "📖 Usage:"
        echo "   1. Start the application: streamlit run frontend/streamlit/app.py"
        echo "   2. Go to 'Web URLs' tab in the sidebar"
        echo "   3. Enter any web URL to extract and process content"
        echo ""
    else
        echo "⚠️  MCP server installed but test failed (this might be normal)"
        echo "   Try using the application - fallback method will work if needed"
    fi
else
    echo "❌ MCP server installation failed"
    echo "   The application will use fallback method (BeautifulSoup)"
    echo "   Make sure you have the required Python packages:"
    echo "   pip install beautifulsoup4 markdownify requests"
fi

echo ""
echo "🔧 Troubleshooting:"
echo "   - If MCP server fails, the app will automatically use fallback"
echo "   - Check that your firewall allows outbound HTTP/HTTPS connections"
echo "   - Some websites may block automated access"
echo ""
echo "📚 Supported by this setup:"
echo "   ✅ News articles and blog posts"
echo "   ✅ Documentation pages"
echo "   ✅ Wikipedia articles"
echo "   ✅ Most static content websites" 