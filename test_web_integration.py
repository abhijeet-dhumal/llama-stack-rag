#!/usr/bin/env python3
"""
Test script for web content processing integration
Run this to verify the MCP server and web processing functionality
"""

import sys
import subprocess
import requests
from pathlib import Path

def test_mcp_server():
    """Test if MCP server is available"""
    print("🧪 Testing MCP Server...")
    try:
        result = subprocess.run(
            ["npx", "@just-every/mcp-read-website-fast", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ MCP Server available")
            return True
        else:
            print("❌ MCP Server not available")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ MCP Server test failed: {e}")
        return False

def test_python_dependencies():
    """Test if required Python packages are available"""
    print("🐍 Testing Python dependencies...")
    
    try:
        import requests
        print("✅ requests available")
    except ImportError:
        print("❌ requests not available")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("✅ beautifulsoup4 available")
    except ImportError:
        print("❌ beautifulsoup4 not available")
        return False
    
    try:
        from markdownify import markdownify
        print("✅ markdownify available")
    except ImportError:
        print("❌ markdownify not available")
        return False
    
    return True

def test_web_processor():
    """Test the web content processor directly"""
    print("🌐 Testing web content processor...")
    
    try:
        # Add the frontend path to sys.path
        frontend_path = Path(__file__).parent / "frontend" / "streamlit"
        sys.path.insert(0, str(frontend_path))
        
        from core.web_content_processor import WebContentProcessor
        
        processor = WebContentProcessor()
        
        # Test URL validation
        test_url = "https://example.com"
        if processor.is_valid_url(test_url):
            print("✅ URL validation works")
        else:
            print("❌ URL validation failed")
            return False
        
        # Test domain info extraction
        domain_info = processor.get_domain_info(test_url)
        if domain_info.get('domain') == 'example.com':
            print("✅ Domain extraction works")
        else:
            print("❌ Domain extraction failed")
            return False
        
        print("✅ Web content processor ready")
        return True
        
    except Exception as e:
        print(f"❌ Web processor test failed: {e}")
        return False

def test_basic_web_extraction():
    """Test basic web content extraction"""
    print("📄 Testing basic web extraction...")
    
    try:
        # Test with requests + BeautifulSoup fallback
        from bs4 import BeautifulSoup
        import requests
        
        test_url = "https://httpbin.org/html"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            
            if len(text) > 50:
                print("✅ Basic web extraction works")
                return True
            else:
                print("❌ Extracted content too short")
                return False
        else:
            print(f"❌ HTTP request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Basic web extraction failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing RAG LlamaStack Web Content Processing Integration")
    print("=" * 60)
    
    tests = [
        ("Python Dependencies", test_python_dependencies),
        ("MCP Server", test_mcp_server),
        ("Web Processor", test_web_processor),
        ("Basic Web Extraction", test_basic_web_extraction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False
    
    print("-" * 30)
    
    if all_passed:
        print("🎉 All tests passed! Web content processing is ready.")
        print("\n💡 Next steps:")
        print("   1. Start the app: make start")
        print("   2. Go to '🌐 Web URLs' tab in sidebar")
        print("   3. Enter any web URL to test")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   - Run: make setup-mcp")
        print("   - Check: npm install")
        print("   - Install: pip install beautifulsoup4 markdownify requests")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 