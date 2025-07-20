"""
Web Content Processor for RAG LlamaStack
Processes web URLs using MCP server with fallback to direct extraction
"""

import streamlit as st
import requests
import subprocess
import json
import tempfile
import os
import time
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urljoin
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    st.warning("BeautifulSoup4 and markdownify not installed. Web content processing will be limited.")


class WebContentProcessor:
    """Process web content using MCP server with fallback options"""
    
    def __init__(self):
        self.mcp_server_path = None
        self.timeout = 30
        self.max_content_length = 50 * 1024 * 1024  # 50MB limit
        
    def is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def get_domain_info(self, url: str) -> Dict[str, str]:
        """Extract domain information from URL"""
        try:
            parsed = urlparse(url)
            return {
                'domain': parsed.netloc,
                'scheme': parsed.scheme,
                'path': parsed.path,
                'full_url': url
            }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_with_mcp_server(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content using MCP read-website-fast server"""
        try:
            # Try to use npx to run the MCP server
            cmd = [
                'npx', '@just-every/mcp-read-website-fast',
                'read-website',
                '--url', url,
                '--format', 'markdown',
                '--timeout', str(self.timeout)
            ]
            
            with st.spinner(f"🌐 Extracting content from {url} using MCP server..."):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 10,
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0:
                    try:
                        # Parse JSON response
                        response = json.loads(result.stdout)
                        return {
                            'content': response.get('content', ''),
                            'title': response.get('title', ''),
                            'metadata': response.get('metadata', {}),
                            'method': 'mcp_server',
                            'success': True
                        }
                    except json.JSONDecodeError:
                        # If not JSON, treat as raw markdown
                        return {
                            'content': result.stdout,
                            'title': f"Content from {url}",
                            'metadata': {'url': url},
                            'method': 'mcp_server',
                            'success': True
                        }
                else:
                    st.warning(f"MCP server error: {result.stderr}")
                    return None
                    
        except subprocess.TimeoutExpired:
            st.error(f"⏱️ MCP server timeout after {self.timeout} seconds")
            return None
        except FileNotFoundError:
            st.warning("📦 MCP server not found. Using fallback method...")
            return None
        except Exception as e:
            st.warning(f"🔄 MCP server error: {str(e)}. Using fallback method...")
            return None
    
    def extract_with_requests(self, url: str) -> Optional[Dict[str, Any]]:
        """Fallback extraction using requests + BeautifulSoup"""
        if not BS4_AVAILABLE:
            return None
            
        try:
            with st.spinner(f"🔄 Extracting content from {url} using fallback method..."):
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
                
                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                # Check content length
                if len(response.content) > self.max_content_length:
                    raise ValueError(f"Content too large: {len(response.content)} bytes")
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else f"Content from {url}"
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.decompose()
                
                # Find main content
                main_content = None
                for selector in ['main', 'article', '.content', '#content', '.post', '.entry']:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                if not main_content:
                    main_content = soup.find('body') or soup
                
                # Convert to markdown
                markdown_content = md(str(main_content), heading_style="ATX")
                
                # Clean up markdown
                lines = markdown_content.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.isspace():
                        cleaned_lines.append(line)
                
                content = '\n\n'.join(cleaned_lines)
                
                return {
                    'content': content,
                    'title': title,
                    'metadata': {
                        'url': url,
                        'domain': urlparse(url).netloc,
                        'content_length': len(content),
                        'extraction_method': 'requests_beautifulsoup'
                    },
                    'method': 'fallback_requests',
                    'success': True
                }
                
        except requests.exceptions.RequestException as e:
            st.error(f"🌐 Network error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Content extraction error: {str(e)}")
            return None
    
    def process_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Process URL with MCP server and fallback methods"""
        
        # Validate URL
        if not self.is_valid_url(url):
            st.error("❌ Invalid URL format. Please enter a valid HTTP/HTTPS URL.")
            return None
        
        # Get domain info for display
        domain_info = self.get_domain_info(url)
        st.info(f"🔗 Processing: {domain_info.get('domain', url)}")
        
        # Try MCP server first
        result = self.extract_with_mcp_server(url)
        
        # If MCP server fails, try fallback
        if not result and BS4_AVAILABLE:
            result = self.extract_with_requests(url)
        
        if result and result.get('success'):
            # Add processing metadata
            result['processed_at'] = time.time()
            result['source_type'] = 'web_url'
            result['source_url'] = url
            
            # Validate content length
            content = result.get('content', '')
            if len(content) < 100:
                st.warning("⚠️ Content seems too short. The page might not have been processed correctly.")
            
            st.success(f"✅ Successfully extracted {len(content)} characters from {domain_info.get('domain', url)}")
            return result
        
        st.error("❌ Failed to extract content from the URL. Please check the URL and try again.")
        return None


def create_web_content_processor() -> WebContentProcessor:
    """Factory function to create web content processor"""
    return WebContentProcessor()


def process_web_url(url: str) -> Optional[Dict[str, Any]]:
    """Convenience function to process a single URL"""
    processor = create_web_content_processor()
    return processor.process_url(url)


# Example usage and testing
if __name__ == "__main__":
    # Test the processor
    processor = WebContentProcessor()
    
    test_urls = [
        "https://docs.python.org/3/tutorial/",
        "https://streamlit.io/",
        "https://github.com/meta-llama/llama-stack"
    ]
    
    for url in test_urls:
        print(f"\nTesting URL: {url}")
        result = processor.process_url(url)
        if result:
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Content length: {len(result.get('content', ''))}")
            print(f"Method: {result.get('method', 'N/A')}")
        else:
            print("Failed to process URL") 