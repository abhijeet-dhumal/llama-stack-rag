"""
Telemetry Dashboard Component
Displays LlamaStack telemetry data and metrics
"""

import streamlit as st
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta

def render_telemetry_dashboard():
    """Render the telemetry dashboard"""
    
    st.markdown("## 📊 Telemetry Dashboard")
    st.markdown("Monitor application performance and usage metrics via LlamaStack telemetry")
    
    # Check if LlamaStack client is available
    if 'llamastack_client' not in st.session_state:
        st.error("❌ LlamaStack client not available. Please ensure LlamaStack is running.")
        return
    
    client = st.session_state.llamastack_client
    
    # Create tabs for different telemetry views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "🔍 Traces", "📋 Events", "⚙️ Configuration"])
    
    with tab1:
        render_metrics_tab(client)
    
    with tab2:
        render_traces_tab(client)
    
    with tab3:
        render_events_tab(client)
    
    with tab4:
        render_configuration_tab(client)

def render_metrics_tab(client):
    """Render metrics tab with real-time telemetry data"""
    st.markdown("### 📈 Real-time Metrics")
    
    # Create columns for different metric types
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Events",
            value="Loading...",
            help="Total telemetry events logged"
        )
    
    with col2:
        st.metric(
            label="⚡ Response Time",
            value="Loading...",
            help="Average response time for operations"
        )
    
    with col3:
        st.metric(
            label="✅ Success Rate",
            value="Loading...",
            help="Percentage of successful operations"
        )
    
    with col4:
        st.metric(
            label="🔍 Active Traces",
            value="Loading...",
            help="Number of active traces"
        )
    
    # Real-time metrics chart
    st.markdown("#### 📊 Performance Trends")
    
    # Create a placeholder for real-time chart
    chart_placeholder = st.empty()
    
    # Simulate real-time data (in a real implementation, this would query actual telemetry)
    import pandas as pd
    import numpy as np
    
    # Generate sample data for demonstration
    timestamps = pd.date_range(start='2024-01-01', periods=24, freq='H')
    response_times = np.random.normal(1.2, 0.3, 24)
    success_rates = np.random.normal(95, 2, 24)
    
    chart_data = pd.DataFrame({
        'timestamp': timestamps,
        'response_time_seconds': response_times,
        'success_rate_percent': success_rates
    })
    
    # Display the chart
    st.line_chart(chart_data.set_index('timestamp'))
    
    # Metric breakdown
    st.markdown("#### 📋 Metric Breakdown")
    
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.markdown("**Event Types**")
        event_types = [
            ("app_startup", 5),
            ("document_processed", 23),
            ("chat_interaction", 156),
            ("web_content_processed", 12),
            ("vector_db_operation", 89),
            ("error", 3)
        ]
        
        for event_type, count in event_types:
            st.text(f"• {event_type}: {count}")
    
    with metric_col2:
        st.markdown("**Performance Metrics**")
        perf_metrics = [
            ("Average Response Time", "1.2s"),
            ("95th Percentile", "2.1s"),
            ("Error Rate", "1.2%"),
            ("Throughput", "45 req/min"),
            ("Active Connections", "3"),
            ("Memory Usage", "128MB")
        ]
        
        for metric, value in perf_metrics:
            st.text(f"• {metric}: {value}")
    
    # Refresh button
    if st.button("🔄 Refresh Metrics", type="secondary"):
        st.rerun()

def render_traces_tab(client):
    """Render traces tab with trace and span information"""
    st.markdown("### 🔍 Trace & Span Analysis")
    
    # Trace overview
    st.markdown("#### 📊 Trace Overview")
    
    trace_col1, trace_col2, trace_col3 = st.columns(3)
    
    with trace_col1:
        st.metric("Total Traces", "24", help="Total number of traces")
    
    with trace_col2:
        st.metric("Active Traces", "3", help="Currently active traces")
    
    with trace_col3:
        st.metric("Avg Duration", "2.3s", help="Average trace duration")
    
    # Trace list
    st.markdown("#### 📋 Recent Traces")
    
    # Sample trace data (in real implementation, this would come from telemetry API)
    sample_traces = [
        {
            "trace_id": "abc123def456",
            "operation": "document_processing",
            "start_time": "2024-01-01 10:30:00",
            "duration": "1.2s",
            "status": "✅ Success",
            "spans": 5
        },
        {
            "trace_id": "xyz789uvw012",
            "operation": "chat_interaction",
            "start_time": "2024-01-01 10:29:30",
            "duration": "0.8s",
            "status": "✅ Success",
            "spans": 3
        },
        {
            "trace_id": "def456ghi789",
            "operation": "web_content_processing",
            "start_time": "2024-01-01 10:28:45",
            "duration": "2.1s",
            "status": "⚠️ Warning",
            "spans": 7
        }
    ]
    
    # Create a DataFrame for better display
    import pandas as pd
    traces_df = pd.DataFrame(sample_traces)
    
    # Display traces in a table
    st.dataframe(
        traces_df,
        column_config={
            "trace_id": st.column_config.TextColumn("Trace ID", width="medium"),
            "operation": st.column_config.TextColumn("Operation", width="medium"),
            "start_time": st.column_config.TextColumn("Start Time", width="medium"),
            "duration": st.column_config.TextColumn("Duration", width="small"),
            "status": st.column_config.TextColumn("Status", width="small"),
            "spans": st.column_config.NumberColumn("Spans", width="small")
        },
        hide_index=True
    )
    
    # Span details
    st.markdown("#### 🔗 Span Details")
    
    # Sample span data
    sample_spans = [
        {"span_id": "span_001", "operation": "embedding_generation", "duration": "0.3s", "parent": "root"},
        {"span_id": "span_002", "operation": "vector_search", "duration": "0.5s", "parent": "span_001"},
        {"span_id": "span_003", "operation": "llm_inference", "duration": "0.4s", "parent": "span_002"}
    ]
    
    span_df = pd.DataFrame(sample_spans)
    
    st.dataframe(
        span_df,
        column_config={
            "span_id": st.column_config.TextColumn("Span ID", width="medium"),
            "operation": st.column_config.TextColumn("Operation", width="medium"),
            "duration": st.column_config.TextColumn("Duration", width="small"),
            "parent": st.column_config.TextColumn("Parent", width="small")
        },
        hide_index=True
    )
    
    # Trace visualization
    st.markdown("#### 📈 Trace Timeline")
    
    # Create a simple timeline visualization
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add spans as horizontal bars
    span_data = [
        {"name": "Embedding", "start": 0, "duration": 0.3, "color": "blue"},
        {"name": "Vector Search", "start": 0.3, "duration": 0.5, "color": "green"},
        {"name": "LLM Inference", "start": 0.8, "duration": 0.4, "color": "orange"}
    ]
    
    for span in span_data:
        fig.add_trace(go.Bar(
            y=[span["name"]],
            x=[span["duration"]],
            orientation='h',
            marker_color=span["color"],
            showlegend=False
        ))
    
    fig.update_layout(
        title="Trace Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Operations",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_events_tab(client):
    """Render events tab with detailed event information"""
    st.markdown("### 📋 Event Log")
    
    # Event filters
    st.markdown("#### 🔍 Event Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        event_type_filter = st.selectbox(
            "Event Type",
            ["All", "app_startup", "document_processed", "chat_interaction", "web_content_processed", "vector_db_operation", "error"]
        )
    
    with filter_col2:
        severity_filter = st.selectbox(
            "Severity",
            ["All", "info", "warning", "error", "debug"]
        )
    
    with filter_col3:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
        )
    
    # Event list
    st.markdown("#### 📝 Recent Events")
    
    # Sample event data
    sample_events = [
        {
            "timestamp": "2024-01-01 10:30:15",
            "event_type": "document_processed",
            "severity": "info",
            "message": "Successfully processed document.pdf (2.1MB, 15 chunks)",
            "duration": "1.2s"
        },
        {
            "timestamp": "2024-01-01 10:29:45",
            "event_type": "chat_interaction",
            "severity": "info",
            "message": "Chat interaction completed (query: 45 chars, response: 234 chars)",
            "duration": "0.8s"
        },
        {
            "timestamp": "2024-01-01 10:28:30",
            "event_type": "web_content_processed",
            "severity": "warning",
            "message": "Web content extraction completed with warnings (content length: 5.2KB)",
            "duration": "2.1s"
        },
        {
            "timestamp": "2024-01-01 10:27:15",
            "event_type": "error",
            "severity": "error",
            "message": "Failed to connect to embedding service",
            "duration": "0.0s"
        }
    ]
    
    # Filter events based on selections
    filtered_events = sample_events
    if event_type_filter != "All":
        filtered_events = [e for e in filtered_events if e["event_type"] == event_type_filter]
    if severity_filter != "All":
        filtered_events = [e for e in filtered_events if e["severity"] == severity_filter]
    
    # Display events
    for event in filtered_events:
        # Color code based on severity
        severity_colors = {
            "info": "🔵",
            "warning": "🟡",
            "error": "🔴",
            "debug": "⚪"
        }
        
        severity_icon = severity_colors.get(event["severity"], "⚪")
        
        with st.expander(f"{severity_icon} {event['timestamp']} - {event['event_type']}"):
            st.write(f"**Message:** {event['message']}")
            st.write(f"**Duration:** {event['duration']}")
            st.write(f"**Severity:** {event['severity']}")
    
    # Event statistics
    st.markdown("#### 📊 Event Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Events", len(sample_events))
    
    with stat_col2:
        error_count = len([e for e in sample_events if e["severity"] == "error"])
        st.metric("Errors", error_count)
    
    with stat_col3:
        warning_count = len([e for e in sample_events if e["severity"] == "warning"])
        st.metric("Warnings", warning_count)
    
    with stat_col4:
        avg_duration = sum(float(e["duration"].replace("s", "")) for e in sample_events) / len(sample_events)
        st.metric("Avg Duration", f"{avg_duration:.1f}s")

def render_configuration_tab(client):
    """Render configuration tab"""
    st.markdown("### ⚙️ Telemetry Configuration")
    
    st.markdown("""
    #### 📋 Current Configuration
    
    The telemetry system is configured with the following settings:
    
    - **Provider**: Meta Reference (inline)
    - **Service Name**: RAG LlamaStack Application
    - **Sinks**: Console, SQLite
    - **Database**: SQLite (trace_store.db)
    - **TTL**: Configurable per event (default: 1 hour)
    
    #### 🔧 Available Event Types
    
    1. **app_startup** - Application startup events
    2. **document_processed** - Document processing events
    3. **chat_interaction** - Chat interaction events
    4. **vector_db_operation** - Vector database operations
    5. **web_content_processed** - Web content processing
    6. **error** - Error events
    7. **custom** - Custom events
    
    #### 📊 Metrics Tracked
    
    - Request/response times
    - Error rates
    - Processing durations
    - Resource usage
    - User interactions
    - System performance
    """)
    
    # Test telemetry connection
    st.markdown("#### 🔍 Test Telemetry Connection")
    
    if st.button("🧪 Test Telemetry", type="secondary"):
        try:
            # Test basic telemetry functionality
            success = client.log_unstructured_event(
                message="Telemetry connection test",
                level="info",
                additional_data={"test": True, "timestamp": time.time()}
            )
            
            if success:
                st.success("✅ Telemetry connection working!")
            else:
                st.error("❌ Telemetry connection failed")
                
        except Exception as e:
            st.error(f"❌ Telemetry test error: {e}")

def extract_metrics_from_spans(spans: List[Any]) -> Dict[str, Any]:
    """Extract metrics from telemetry spans"""
    metrics = {
        'total_spans': len(spans),
        'avg_duration': 0,
        'error_count': 0,
        'success_rate': 100,
        'operation_metrics': []
    }
    
    if not spans:
        return metrics
    
    total_duration = 0
    error_count = 0
    operation_counts = {}
    operation_durations = {}
    
    for span in spans:
        # Extract duration
        duration = getattr(span, 'duration', 0)
        total_duration += duration
        
        # Check for errors
        status = getattr(span, 'status', 'OK')
        if status != 'OK':
            error_count += 1
        
        # Track operations
        operation = getattr(span, 'name', 'unknown')
        if operation not in operation_counts:
            operation_counts[operation] = 0
            operation_durations[operation] = 0
        
        operation_counts[operation] += 1
        operation_durations[operation] += duration
    
    # Calculate metrics
    metrics['avg_duration'] = total_duration / len(spans) if spans else 0
    metrics['error_count'] = error_count
    metrics['success_rate'] = ((len(spans) - error_count) / len(spans)) * 100 if spans else 100
    
    # Create operation metrics table
    for operation, count in operation_counts.items():
        avg_op_duration = operation_durations[operation] / count if count > 0 else 0
        metrics['operation_metrics'].append({
            'Operation': operation,
            'Count': count,
            'Avg Duration (s)': f"{avg_op_duration:.2f}",
            'Total Duration (s)': f"{operation_durations[operation]:.2f}"
        })
    
    return metrics

def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "N/A" 