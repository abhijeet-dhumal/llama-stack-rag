#!/usr/bin/env python3
"""
Simple test to verify telemetry functionality
"""

import time
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.event_param import UnstructuredLogEvent, MetricEvent, StructuredLogEvent

def test_basic_telemetry():
    """Test basic telemetry functionality"""
    print("🚀 Testing Basic Telemetry Functionality")
    print("=" * 50)
    
    try:
        # Initialize client
        client = LlamaStackClient()
        
        # Test 1: Check if telemetry is available
        print("✅ Telemetry client available:", hasattr(client, 'telemetry'))
        
        # Test 2: Log a simple unstructured event
        print("📝 Testing unstructured event...")
        try:
            event = UnstructuredLogEvent({
                "message": "Test unstructured log event",
                "level": "info",
                "timestamp": time.time(),
                "source": "test_script"
            })
            
            client.telemetry.log_event(
                event=event,
                ttl_seconds=3600
            )
            print("✅ Unstructured event logged successfully!")
            
        except Exception as e:
            print(f"❌ Error with unstructured event: {e}")
        
        # Test 3: Log a metric event
        print("📊 Testing metric event...")
        try:
            event = MetricEvent({
                "metric_name": "test_metric",
                "value": 42.0,
                "timestamp": time.time(),
                "tags": {"source": "test_script", "type": "counter"}
            })
            
            client.telemetry.log_event(
                event=event,
                ttl_seconds=3600
            )
            print("✅ Metric event logged successfully!")
            
        except Exception as e:
            print(f"❌ Error with metric event: {e}")
        
        # Test 4: Log a structured event
        print("📋 Testing structured event...")
        try:
            event = StructuredLogEvent({
                "event_type": "test_structured",
                "data": {
                    "user_id": "test_user",
                    "action": "test_action",
                    "timestamp": time.time()
                },
                "metadata": {
                    "version": "1.0.0",
                    "environment": "test"
                }
            })
            
            client.telemetry.log_event(
                event=event,
                ttl_seconds=3600
            )
            print("✅ Structured event logged successfully!")
            
        except Exception as e:
            print(f"❌ Error with structured event: {e}")
        
        print("\n" + "=" * 50)
        print("🏁 Basic telemetry test completed!")
        
    except Exception as e:
        print(f"❌ Error in basic telemetry test: {e}")

if __name__ == "__main__":
    test_basic_telemetry() 