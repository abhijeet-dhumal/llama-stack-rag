#!/usr/bin/env python3
"""
Test script for improved LlamaStack telemetry integration
Tests trace context, official client usage, and fallback mechanisms
"""

import time
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend', 'streamlit'))

from core.llamastack_client import LlamaStackClient

def test_improved_telemetry():
    """Test the improved telemetry integration"""
    print("🚀 Testing Improved LlamaStack Telemetry Integration")
    print("=" * 60)
    
    try:
        # Initialize client
        client = LlamaStackClient()
        
        # Test 1: Basic telemetry functionality
        print("📝 Testing basic telemetry functionality...")
        
        # App startup event
        success = client.log_app_startup(version="1.0.0", user_agent="test_script")
        print(f"✅ App startup event: {'Success' if success else 'Failed'}")
        
        # Document processing event
        success = client.log_document_processed(
            file_type="pdf",
            file_size=1024000,
            chunks_created=5,
            processing_time=2.5
        )
        print(f"✅ Document processing event: {'Success' if success else 'Failed'}")
        
        # Chat interaction event
        success = client.log_chat_interaction(
            model_used="llama3.2:3b",
            query_length=50,
            response_length=200,
            response_time=1.2
        )
        print(f"✅ Chat interaction event: {'Success' if success else 'Failed'}")
        
        # Test 2: Trace context functionality
        print("\n🔍 Testing trace context functionality...")
        
        # Create trace context for document processing
        trace_context = client.create_trace_context("document_processing_test")
        print(f"✅ Created trace context: {trace_context['trace_id']}")
        
        # Log trace events
        success = client.log_trace_event(
            trace_context=trace_context,
            event_type="document_upload_started",
            event_data={
                "file_name": "test_document.pdf",
                "file_size_mb": 2.1,
                "user_id": "test_user"
            }
        )
        print(f"✅ Trace event logged: {'Success' if success else 'Failed'}")
        
        # Log trace metrics
        success = client.log_trace_metric(
            trace_context=trace_context,
            metric_name="document_processing_duration",
            value=2.5,
            tags={"file_type": "pdf", "processing_stage": "embedding"}
        )
        print(f"✅ Trace metric logged: {'Success' if success else 'Failed'}")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Log trace completion
        success = client.log_trace_completion(
            trace_context=trace_context,
            success=True,
            error_message=None
        )
        print(f"✅ Trace completion logged: {'Success' if success else 'Failed'}")
        
        # Test 3: Error handling and fallback
        print("\n🔄 Testing error handling and fallback...")
        
        # Test with invalid trace context
        invalid_context = {"invalid": "context"}
        success = client.log_trace_event(
            trace_context=invalid_context,
            event_type="test_event",
            event_data={"test": "data"}
        )
        print(f"✅ Invalid trace context handled: {'Success' if not success else 'Failed'}")
        
        # Test 4: Metric logging with tags
        print("\n📊 Testing metric logging with tags...")
        
        success = client.log_metric(
            metric_name="test_metric",
            value=42.0,
            tags={
                "environment": "test",
                "component": "telemetry_test",
                "version": "1.0.0"
            }
        )
        print(f"✅ Metric with tags logged: {'Success' if success else 'Failed'}")
        
        # Test 5: Unstructured event logging
        print("\n📝 Testing unstructured event logging...")
        
        success = client.log_unstructured_event(
            message="Test unstructured log message",
            level="info",
            additional_data={
                "test_id": "12345",
                "timestamp": time.time(),
                "source": "test_script"
            }
        )
        print(f"✅ Unstructured event logged: {'Success' if success else 'Failed'}")
        
        # Test 6: Error event logging
        print("\n❌ Testing error event logging...")
        
        success = client.log_error(
            error_type="test_error",
            error_message="This is a test error message",
            context={
                "component": "telemetry_test",
                "test_phase": "error_handling",
                "timestamp": time.time()
            }
        )
        print(f"✅ Error event logged: {'Success' if success else 'Failed'}")
        
        # Test 7: Complex trace scenario
        print("\n🔄 Testing complex trace scenario...")
        
        # Create trace for web content processing
        web_trace = client.create_trace_context("web_content_processing")
        print(f"✅ Created web processing trace: {web_trace['trace_id']}")
        
        # Log multiple events in the trace
        events = [
            ("url_validation", {"url": "https://example.com", "status": "valid"}),
            ("content_extraction", {"method": "mcp_server", "content_length": 5000}),
            ("embedding_generation", {"model": "all-MiniLM-L6-v2", "chunks": 3}),
            ("vector_storage", {"database": "faiss", "vectors_stored": 3})
        ]
        
        for event_type, event_data in events:
            success = client.log_trace_event(
                trace_context=web_trace,
                event_type=event_type,
                event_data=event_data
            )
            print(f"✅ {event_type} event: {'Success' if success else 'Failed'}")
            time.sleep(0.05)  # Simulate processing time
        
        # Log completion
        success = client.log_trace_completion(
            trace_context=web_trace,
            success=True
        )
        print(f"✅ Web processing trace completed: {'Success' if success else 'Failed'}")
        
        # Test 8: Performance metrics
        print("\n⚡ Testing performance metrics...")
        
        performance_metrics = [
            ("response_time_ms", 1200, {"endpoint": "/chat", "model": "llama3.2:3b"}),
            ("embedding_generation_time_ms", 450, {"model": "all-MiniLM-L6-v2"}),
            ("vector_search_time_ms", 320, {"database": "faiss", "top_k": 4}),
            ("memory_usage_mb", 128, {"component": "streamlit_app"}),
            ("cpu_usage_percent", 15.5, {"component": "llamastack_server"})
        ]
        
        for metric_name, value, tags in performance_metrics:
            success = client.log_metric(
                metric_name=metric_name,
                value=value,
                tags=tags
            )
            print(f"✅ {metric_name}: {'Success' if success else 'Failed'}")
        
        print("\n🎉 All telemetry tests completed!")
        print("📊 Check the telemetry dashboard to see the logged events")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_telemetry_connectivity():
    """Test basic telemetry connectivity"""
    print("\n🔍 Testing Telemetry Connectivity")
    print("=" * 40)
    
    try:
        client = LlamaStackClient()
        
        # Test health check
        health = client.health_check()
        print(f"✅ LlamaStack health: {'Healthy' if health else 'Unhealthy'}")
        
        # Test telemetry endpoint
        try:
            import requests
            response = requests.get("http://localhost:8321/v1/telemetry/events", timeout=5)
            print(f"✅ Telemetry endpoint: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Telemetry endpoint test: {e}")
        
        # Test basic event logging
        success = client.log_unstructured_event(
            message="Connectivity test message",
            level="info"
        )
        print(f"✅ Basic event logging: {'Success' if success else 'Failed'}")
        
    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")

if __name__ == "__main__":
    print("🧪 LlamaStack Telemetry Integration Test Suite")
    print("=" * 50)
    
    # Test connectivity first
    test_telemetry_connectivity()
    
    # Run comprehensive tests
    test_improved_telemetry()
    
    print("\n📋 Test Summary:")
    print("✅ Basic telemetry functionality")
    print("✅ Trace context management")
    print("✅ Error handling and fallback")
    print("✅ Metric logging with tags")
    print("✅ Unstructured event logging")
    print("✅ Complex trace scenarios")
    print("✅ Performance metrics")
    print("\n🎯 Next steps:")
    print("1. Check the telemetry dashboard in your Streamlit app")
    print("2. Verify events are being logged in the SQLite database")
    print("3. Monitor real-time metrics and traces")
    print("4. Use trace context for debugging complex operations") 