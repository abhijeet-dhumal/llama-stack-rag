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
        """Check if URL is valid and well-formed"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def validate_github_url(self, url: str) -> Dict[str, Any]:
        """Validate and provide feedback for GitHub URLs"""
        if 'github.com' not in url:
            return {'valid': True, 'message': None}
        
        # Check if it's just a domain
        if url.strip() in ['github.com', 'https://github.com', 'http://github.com']:
            return {
                'valid': False, 
                'message': "❌ Please enter a specific GitHub repository URL, not just 'github.com'.\n\n**Examples:**\n- https://github.com/kubeflow/trainer/blob/master/README.md\n- https://github.com/username/repo-name\n- https://github.com/username/repo-name/blob/main/README.md"
            }
        
        # Check if it has a repository path
        if '/blob/' in url or '/tree/' in url:
            return {'valid': True, 'message': None}
        
        # Check if it's a repository root (no specific file)
        if url.count('/') >= 4:  # github.com/username/repo
            return {
                'valid': True, 
                'message': "ℹ️ Repository URL detected. Will try to find README.md automatically."
            }
        
        return {
            'valid': False, 
            'message': "❌ Invalid GitHub URL format.\n\n**Valid formats:**\n- https://github.com/username/repo-name\n- https://github.com/username/repo-name/blob/main/README.md\n- https://github.com/username/repo-name/blob/main/folder/file.md"
        }
    
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
        """Extract content using multiple MCP server options with intelligent fallback"""
        
        mcp_servers = [
            {
                'name': 'just-every',
                'command': ['npx', '@just-every/mcp-read-website-fast', 'fetch'],
                'args': ['--output', 'markdown'],
                'description': 'Just-Every MCP (reliable markdown extraction)',
                'requires_api_key': False
            }
        ]
        
        for server in mcp_servers:
            try:
                # Check if API key is required but not available
                if server.get('requires_api_key', False):
                    api_key_env = server.get('api_key_env', '')
                    if not os.environ.get(api_key_env):
                        print(f"⚠️ {server['description']} requires {api_key_env} environment variable")
                        continue
                
                # Skip server-based MCPs for now (they need different handling)
                if server.get('is_server', False):
                    print(f"⚠️ {server['description']} is a server-based MCP (skipping for now)")
                    continue
                
                print(f"🔄 Trying {server['description']}...")
                with st.spinner(f"🌐 Extracting content from {url} using {server['description']}..."):
                    cmd = server['command'] + [url] + server['args'] + ['--timeout', str(self.timeout * 1000)]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout + 10,
                        cwd=os.getcwd()
                    )
                    
                    if result.returncode == 0:
                        try:
                            # Try to parse JSON response first
                            response = json.loads(result.stdout)
                            return {
                                'content': response.get('content', ''),
                                'title': response.get('title', ''),
                                'metadata': response.get('metadata', {}),
                                'method': f"mcp_{server['name']}",
                                'success': True,
                                'server_used': server['description']
                            }
                        except json.JSONDecodeError:
                            # If not JSON, treat as raw markdown
                            content = result.stdout.strip()
                            if len(content) > 100:  # Ensure we got meaningful content
                                # Check for common poor content indicators
                                poor_content_indicators = [
                                    'navigation menu', 'skip to content', 'sign in', 
                                    'appearance settings', 'product', 'features',
                                    'toggle navigation', 'github copilot', 'github models',
                                    'trainer/readme.md at master', 'by kubeflow'
                                ]
                                
                                content_lower = content.lower()
                                poor_content_score = sum(1 for indicator in poor_content_indicators if indicator in content_lower)
                                
                                # For GitHub URLs, be more strict about content quality
                                if 'github.com' in url and poor_content_score > 2:
                                    print(f"⚠️ {server['description']} returned GitHub navigation content, trying fallback...")
                                    continue
                                elif poor_content_score > 3:
                                    print(f"⚠️ {server['description']} returned poor content (navigation/menu), trying next...")
                                    continue
                                
                                return {
                                    'content': content,
                                    'title': f"Content from {url}",
                                    'metadata': {'url': url, 'extraction_method': server['name']},
                                    'method': f"mcp_{server['name']}",
                                    'success': True,
                                    'server_used': server['description']
                                }
                            else:
                                # Content too short, try next server
                                print(f"⚠️ {server['description']} returned too little content ({len(content)} chars), trying next...")
                                continue
                    else:
                        # Log error but continue to next server
                        print(f"⚠️ {server['description']} failed: {result.stderr}")
                        continue
                        
            except subprocess.TimeoutExpired:
                print(f"⏱️ {server['description']} timeout after {self.timeout} seconds")
                continue
            except FileNotFoundError:
                print(f"📦 {server['description']} not found, trying next...")
                continue
            except Exception as e:
                print(f"🔄 {server['description']} error: {str(e)}, trying next...")
                continue
        
        # If all MCP servers failed
        print("❌ All MCP servers failed, trying fallback...")
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
    
    def extract_with_fallback(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content using fallback methods when MCP servers fail"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from markdownify import markdownify

            # Special handling for GitHub repository root URLs
            if 'github.com' in url and '/blob/' not in url and url.count('/') >= 4:
                # Try to find README.md automatically
                readme_urls = [
                    url + '/blob/main/README.md',
                    url + '/blob/master/README.md',
                    url + '/blob/main/readme.md',
                    url + '/blob/master/readme.md'
                ]
                
                for readme_url in readme_urls:
                    raw_url = readme_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                    print(f"🔄 Trying README at: {raw_url}")
                    
                    try:
                        response = requests.get(raw_url, timeout=self.timeout)
                        if response.status_code == 200:
                            content = response.text
                            if len(content) > 100:  # Ensure we got meaningful content
                                return {
                                    'content': content,
                                    'title': f"README.md from {url}",
                                    'metadata': {'url': url, 'raw_url': raw_url, 'content_type': 'text/plain'},
                                    'method': 'requests_github_raw',
                                    'success': True,
                                    'server_used': 'Direct HTTP request (GitHub raw content)'
                                }
                    except Exception as e:
                        print(f"🔄 README attempt failed: {str(e)}")
                        continue

            # Special handling for GitHub URLs - try raw URL for better content
            if 'github.com' in url and '/blob/' in url:
                raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                print(f"🔄 Trying raw GitHub URL for better content: {url} → {raw_url}")

                try:
                    response = requests.get(raw_url, timeout=self.timeout)
                    response.raise_for_status()

                    if 'text/plain' in response.headers.get('content-type', '').lower():
                        content = response.text
                        title = f"Content from {url}"
                        return {
                            'content': content,
                            'title': title,
                            'metadata': {'url': url, 'raw_url': raw_url, 'content_type': 'text/plain'},
                            'method': 'requests_github_raw',
                            'success': True,
                            'server_used': 'Direct HTTP request (GitHub raw content)'
                        }
                except Exception as e:
                    print(f"🔄 Raw GitHub URL failed: {str(e)}, trying original URL...")

            # Try direct request with original URL
            response = requests.get(url, timeout=self.timeout)
            
            # Handle specific HTTP status codes
            if response.status_code == 404:
                return {
                    'content': f"Page not found: {url}",
                    'title': f"Error - Page Not Found",
                    'metadata': {'url': url, 'status_code': 404},
                    'method': 'requests_error',
                    'success': False,
                    'error': 'HTTP 404 - Page not found',
                    'server_used': 'Direct HTTP request'
                }
            elif response.status_code == 403:
                return {
                    'content': f"Access forbidden: {url}",
                    'title': f"Error - Access Forbidden",
                    'metadata': {'url': url, 'status_code': 403},
                    'method': 'requests_error',
                    'success': False,
                    'error': 'HTTP 403 - Access forbidden',
                    'server_used': 'Direct HTTP request'
                }
            elif response.status_code >= 500:
                return {
                    'content': f"Server error: {url}",
                    'title': f"Error - Server Error",
                    'metadata': {'url': url, 'status_code': response.status_code},
                    'method': 'requests_error',
                    'success': False,
                    'error': f'HTTP {response.status_code} - Server error',
                    'server_used': 'Direct HTTP request'
                }
            
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()

            if 'text/plain' in content_type or 'text/markdown' in content_type:
                # Raw text content
                content = response.text
                title = f"Content from {url}"
                return {
                    'content': content,
                    'title': title,
                    'metadata': {'url': url, 'content_type': content_type},
                    'method': 'requests_plain_text',
                    'success': True,
                    'server_used': 'Direct HTTP request (plain text)'
                }
            elif 'text/html' in content_type:
                # HTML content - use BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get title
                title_tag = soup.find('title')
                title = title_tag.get_text() if title_tag else f"Content from {url}"

                # Convert to markdown
                content = markdownify(str(soup), heading_style="ATX")

                return {
                    'content': content,
                    'title': title,
                    'metadata': {'url': url, 'content_type': content_type},
                    'method': 'requests_beautifulsoup',
                    'success': True,
                    'server_used': 'BeautifulSoup (HTML parsing)'
                }
            else:
                # Try to extract text content even for unsupported content types
                try:
                    content = response.text
                    if len(content) > 100:  # Only return if we got meaningful content
                        return {
                            'content': content,
                            'title': f"Content from {url}",
                            'metadata': {'url': url, 'content_type': content_type},
                            'method': 'requests_plain_text',
                            'success': True,
                            'server_used': f'Direct HTTP request ({content_type})'
                        }
                except Exception as e:
                    print(f"🔄 Failed to extract text from {content_type}: {str(e)}")
                
                # If we get here, we couldn't extract meaningful content
                return {
                    'content': f"Unable to extract content from {url}. Content type: {content_type}",
                    'title': f"Content from {url}",
                    'metadata': {'url': url, 'content_type': content_type, 'error': 'Unsupported content type'},
                    'method': 'requests_fallback',
                    'success': False,
                    'error': f'Unsupported content type: {content_type}',
                    'server_used': 'Direct HTTP request (fallback)'
                }

        except Exception as e:
            print(f"🔄 Fallback extraction failed: {str(e)}")
            return None
    
    def process_url(self, url: str, silent_mode: bool = False) -> Optional[Dict[str, Any]]:
        """Process URL with MCP server and fallback methods"""
        try:
            # Validate URL
            if not self.is_valid_url(url):
                if not silent_mode:
                    st.error("❌ Invalid URL format. Please enter a valid HTTP/HTTPS URL.")
                return None
            
            # Validate GitHub URLs specifically
            github_validation = self.validate_github_url(url)
            if not github_validation['valid']:
                if not silent_mode:
                    st.error(github_validation['message'])
                return None
            
            # Show GitHub-specific info if applicable (only in non-silent mode)
            if not silent_mode and github_validation['message'] and 'Repository URL detected' in github_validation['message']:
                st.info(github_validation['message'])
            
            # Get domain info for display (only in non-silent mode)
            domain_info = self.get_domain_info(url)
            if not silent_mode:
                st.info(f"🔗 Processing: {domain_info.get('domain', url)}")
            
            # Try MCP server first
            if silent_mode:
                print(f"🔄 Trying MCP server for {url}")
            result = self.extract_with_mcp_server(url)
            if result and result.get('success'):
                # Add processing metadata
                result['processed_at'] = time.time()
                result['source_type'] = 'web_url'
                result['source_url'] = url
                
                # Validate content length
                content = result.get('content', '')
                if len(content) < 100 and not silent_mode:
                    st.warning("⚠️ Content seems too short. The page might not have been processed correctly.")
                
                # Show success message only in non-silent mode
                if not silent_mode:
                    method = result.get('method', 'unknown')
                    server_used = result.get('server_used', 'Unknown')
                    success_msg = f"✅ Successfully extracted {len(content)} characters from {domain_info.get('domain', url)}"
                    
                    if method.startswith('mcp_'):
                        st.success(success_msg)
                        st.info(f"🔧 **MCP Server Used:** {server_used}")
                    elif method.startswith('requests_'):
                        st.success(success_msg)
                        if method == 'requests_github_raw':
                            st.info(f"🔄 **Method:** GitHub Raw Content")
                        else:
                            st.info(f"🔄 **Method:** {server_used}")
                    else:
                        st.success(success_msg)
                        st.info(f"📊 **Method:** {method}")
                
                return result
            
            # If MCP server failed, try fallback methods (only log in silent mode)
            if silent_mode:
                print(f"📦 MCP server failed for {url}, trying fallback methods...")
            
            result = self.extract_with_fallback(url)
            if result and result.get('success'):
                # Add processing metadata
                result['processed_at'] = time.time()
                result['source_type'] = 'web_url'
                result['source_url'] = url
                
                # Validate content length
                content = result.get('content', '')
                if len(content) < 100 and not silent_mode:
                    st.warning("⚠️ Content seems too short. The page might not have been processed correctly.")
                
                # Show fallback method information only in non-silent mode
                if not silent_mode:
                    method = result.get('method', 'unknown')
                    server_used = result.get('server_used', 'Unknown')
                    success_msg = f"✅ Successfully extracted {len(content)} characters from {domain_info.get('domain', url)}"
                    
                    if method == 'requests_github_raw':
                        st.success(success_msg)
                        st.info(f"🔄 **Method:** GitHub Raw Content")
                    elif method.startswith('requests_'):
                        st.success(success_msg)
                        st.info(f"🔄 **Method:** {server_used}")
                    else:
                        st.success(success_msg)
                        st.info(f"📊 **Method:** {method}")
                
                return result
            
            # If all methods failed
            if silent_mode:
                print(f"❌ All extraction methods failed for {url}")
            
            if not silent_mode:
                st.error(f"❌ Failed to extract content from {domain_info.get('domain', url)}")
                st.info("💡 **Troubleshooting:**\n- Check if the URL is accessible\n- Try a different URL\n- The page might require JavaScript or authentication")
            
            return None
            
        except Exception as e:
            if silent_mode:
                print(f"❌ Exception during processing {url}: {str(e)}")
            else:
                st.error(f"❌ Error processing {url}: {str(e)}")
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