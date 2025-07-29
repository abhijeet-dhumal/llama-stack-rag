#!/usr/bin/env python3
"""
Test telemetry using direct HTTP requests
"""

import time
import requests
import hashlib
import uuid

def generate_hex_id() -> str:
    """Generate a hex ID for telemetry"""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16]

def test_direct_telemetry():
    """Test telemetry using direct HTTP requests"""
    print("🚀 Testing Direct Telemetry API")
    print("=" * 50)
    
    base_url = "http://localhost:8321/v1"
    
    # Test 1: Unstructured log event
    print("📝 Testing unstructured event...")
    try:
        event_data = {
            "trace_id": generate_hex_id(),
            "span_id": generate_hex_id(),
            "timestamp": time.time(),
            "message": "Test unstructured log event",
            "severity": "info"
        }
        
        response = requests.post(
            f"{base_url}/telemetry/events",
            json={
                "event": event_data,
                "ttl_seconds": 3600
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Unstructured event logged successfully!")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error with unstructured event: {e}")
    
    # Test 2: Metric event
    print("📊 Testing metric event...")
    try:
        event_data = {
            "trace_id": generate_hex_id(),
            "span_id": generate_hex_id(),
            "timestamp": time.time(),
            "metric": "test_metric",
            "value": 42.0,
            "unit": "count"
        }
        
        response = requests.post(
            f"{base_url}/telemetry/events",
            json={
                "event": event_data,
                "ttl_seconds": 3600
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Metric event logged successfully!")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error with metric event: {e}")
    
    # Test 3: Structured event
    print("📋 Testing structured event...")
    try:
        event_data = {
            "trace_id": generate_hex_id(),
            "span_id": generate_hex_id(),
            "timestamp": time.time(),
            "payload": {
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
            }
        }
        
        response = requests.post(
            f"{base_url}/telemetry/events",
            json={
                "event": event_data,
                "ttl_seconds": 3600
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Structured event logged successfully!")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error with structured event: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Direct telemetry test completed!")

if __name__ == "__main__":
    test_direct_telemetry() 