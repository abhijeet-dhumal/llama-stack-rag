#!/usr/bin/env python3
"""
Test the updated telemetry integration
"""

import time
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend', 'streamlit'))

from core.llamastack_client import LlamaStackClient

def test_updated_telemetry():
    """Test the updated telemetry integration"""
    print("🚀 Testing Updated Telemetry Integration")
    print("=" * 50)
    
    try:
        # Initialize client
        client = LlamaStackClient()
        
        # Test 1: App startup event
        print("📝 Testing app startup event...")
        success = client.log_app_startup(version="1.0.0", user_agent="test_script")
        print(f"✅ App startup event: {'Success' if success else 'Failed'}")
        
        # Test 2: Document processing event
        print("📄 Testing document processing event...")
        success = client.log_document_processed(
            file_type="pdf",
            file_size=1024000,
            chunks_created=5,
            processing_time=2.5
        )
        print(f"✅ Document processing event: {'Success' if success else 'Failed'}")
        
        # Test 3: Chat interaction event
        print("💬 Testing chat interaction event...")
        success = client.log_chat_interaction(
            model_used="llama3.2:3b",
            query_length=50,
            response_length=200,
            response_time=1.2
        )
        print(f"✅ Chat interaction event: {'Success' if success else 'Failed'}")
        
        # Test 4: Web content processing event
        print("🌐 Testing web content processing event...")
        success = client.log_web_content_processed(
            url="https://example.com",
            content_length=5000,
            processing_time=1.0,
            chunks_created=3
        )
        print(f"✅ Web content processing event: {'Success' if success else 'Failed'}")
        
        # Test 5: Vector DB operation event
        print("🗄️ Testing vector DB operation event...")
        success = client.log_vector_db_operation(
            operation="insert",
            vector_db_id="faiss",
            vectors_count=10,
            operation_time=0.5
        )
        print(f"✅ Vector DB operation event: {'Success' if success else 'Failed'}")
        
        # Test 6: Error event
        print("❌ Testing error event...")
        success = client.log_error(
            error_type="test_error",
            error_message="This is a test error",
            context={"component": "test_script"}
        )
        print(f"✅ Error event: {'Success' if success else 'Failed'}")
        
        # Test 7: Custom metric
        print("📊 Testing custom metric...")
        success = client.log_metric(
            metric_name="test_metric",
            value=42.0,
            tags={"source": "test_script", "type": "counter"}
        )
        print(f"✅ Custom metric: {'Success' if success else 'Failed'}")
        
        # Test 8: Unstructured event
        print("📝 Testing unstructured event...")
        success = client.log_unstructured_event(
            message="Test unstructured log message",
            level="info",
            additional_data={"test": True, "timestamp": time.time()}
        )
        print(f"✅ Unstructured event: {'Success' if success else 'Failed'}")
        
        print("\n" + "=" * 50)
        print("🏁 Updated telemetry integration test completed!")
        print("💡 Check the telemetry dashboard in the Streamlit app to see these events")
        
    except Exception as e:
        print(f"❌ Error in updated telemetry test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_telemetry() 