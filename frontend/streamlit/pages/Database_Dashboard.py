"""
SQLite Database Dashboard for RAG LlamaStack
Provides a comprehensive view of the database with tables, data, and management capabilities
"""

import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Database Dashboard - RAG LlamaStack",
    page_icon="🗄️",
    layout="wide"
)

def get_db_path():
    """Get the database file path"""
    data_dir = Path("data")
    return data_dir / "rag_llamastack.db"

def get_db_connection():
    """Get a connection to the SQLite database"""
    db_path = get_db_path()
    if not db_path.exists():
        st.error("❌ Database file not found. Please ensure the application has been initialized.")
        return None
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except Exception as e:
        st.error(f"❌ Error connecting to database: {e}")
        return None

def get_table_info(conn):
    """Get information about all tables in the database"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in cursor.fetchall()]
        
        table_info = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            table_info[table] = columns
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            row_count = cursor.fetchone()['count']
            table_info[f"{table}_count"] = row_count
            
        return table_info
    except Exception as e:
        st.error(f"❌ Error getting table info: {e}")
        return {}

def get_table_data(conn, table_name, limit=100):
    """Get data from a specific table"""
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"❌ Error reading table {table_name}: {e}")
        return pd.DataFrame()

def execute_custom_query(conn, query):
    """Execute a custom SQL query"""
    try:
        df = pd.read_sql_query(query, conn)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def main():
    """Main database dashboard page"""
    
    # Check authentication
    try:
        from core.auth.authentication import AuthManager
        if not AuthManager.is_authenticated():
            st.error("🔐 Please log in to access the database dashboard")
            return
    except ImportError:
        st.warning("⚠️ Authentication module not available")
    
    # Page header
    st.title("🗄️ Database Dashboard")
    st.markdown("Comprehensive view of the SQLite database with tables, data, and management capabilities")
    
    # Back button
    if st.button("← Back to Main", type="secondary"):
        st.session_state.current_page = None
        st.rerun()
    
    st.markdown("---")
    
    # Get database connection
    conn = get_db_connection()
    if not conn:
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📋 Tables", "🔍 Query", "📈 Analytics", "⚙️ Management"])
    
    with tab1:
        # Database Overview
        st.markdown("### 📊 Database Overview")
        
        # Database file info
        db_path = get_db_path()
        if db_path.exists():
            file_size = db_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(db_path.stat().st_mtime)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Database Size", f"{file_size_mb:.2f} MB")
            with col2:
                st.metric("Last Modified", modified_time.strftime("%Y-%m-%d"))
            with col3:
                st.metric("File Path", str(db_path))
            with col4:
                st.metric("Status", "🟢 Connected")
        
        # Get table information
        table_info = get_table_info(conn)
        
        if table_info:
            st.markdown("### 📋 Database Tables")
            
            # Create a summary dataframe
            table_summary = []
            for table_name in [k for k in table_info.keys() if not k.endswith('_count')]:
                row_count = table_info.get(f"{table_name}_count", 0)
                column_count = len(table_info[table_name])
                table_summary.append({
                    "Table": table_name,
                    "Rows": row_count,
                    "Columns": column_count,
                    "Size (est)": f"{row_count * column_count * 100:.0f} bytes"
                })
            
            if table_summary:
                df_summary = pd.DataFrame(table_summary)
                st.dataframe(df_summary, use_container_width=True)
                
                # Show total statistics
                total_rows = sum(row['Rows'] for row in table_summary)
                total_tables = len(table_summary)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tables", total_tables)
                with col2:
                    st.metric("Total Rows", total_rows)
        else:
            st.warning("⚠️ No tables found in the database")
    
    with tab2:
        # Tables Detail
        st.markdown("### 📋 Table Details")
        
        table_info = get_table_info(conn)
        if not table_info:
            st.warning("⚠️ No tables found")
            return
        
        # Table selector
        table_names = [k for k in table_info.keys() if not k.endswith('_count')]
        selected_table = st.selectbox("Select Table", table_names)
        
        if selected_table:
            st.markdown(f"#### 📄 {selected_table}")
            
            # Show table schema
            st.markdown("**Schema:**")
            columns_info = table_info[selected_table]
            schema_df = pd.DataFrame([
                {
                    "Column": col['name'],
                    "Type": col['type'],
                    "Not Null": "Yes" if col['notnull'] else "No",
                    "Primary Key": "Yes" if col['pk'] else "No",
                    "Default": col['dflt_value'] if col['dflt_value'] else "None"
                }
                for col in columns_info
            ])
            st.dataframe(schema_df, use_container_width=True)
            
            # Show table data
            st.markdown("**Data Preview:**")
            row_limit = st.slider("Rows to display", 10, 1000, 100)
            df_data = get_table_data(conn, selected_table, row_limit)
            
            if not df_data.empty:
                st.dataframe(df_data, use_container_width=True)
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{selected_table}.csv",
                        mime="text/csv"
                    )
                with col2:
                    json_str = df_data.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"{selected_table}.json",
                        mime="application/json"
                    )
            else:
                st.info("📭 No data found in this table")
    
    with tab3:
        # Custom Query
        st.markdown("### 🔍 Custom SQL Query")
        
        # Query input
        query = st.text_area(
            "Enter SQL Query",
            placeholder="SELECT * FROM users LIMIT 10;",
            height=150
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Execute Query", type="primary"):
                if query.strip():
                    df_result, error = execute_custom_query(conn, query)
                    if error:
                        st.error(f"❌ Query Error: {error}")
                    else:
                        st.success(f"✅ Query executed successfully! Found {len(df_result)} rows")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Export results
                        if not df_result.empty:
                            csv = df_result.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.warning("⚠️ Please enter a SQL query")
        
        with col2:
            if st.button("🗑️ Clear"):
                st.rerun()
        
        # Query examples
        st.markdown("#### 💡 Query Examples")
        examples = {
            "Users": "SELECT * FROM users LIMIT 10;",
            "Documents by User": "SELECT u.username, d.name, d.file_type FROM documents d JOIN users u ON d.user_id = u.id;",
            "Document Count": "SELECT file_type, COUNT(*) as count FROM documents GROUP BY file_type;",
            "Recent Uploads": "SELECT name, file_type, upload_time FROM documents ORDER BY upload_time DESC LIMIT 10;",
            "Delete All Documents": "DELETE FROM documents;",
            "Delete All Document Chunks": "DELETE FROM document_chunks;",
            "Reset Document IDs": "DELETE FROM sqlite_sequence WHERE name='documents';"
        }
        
        selected_example = st.selectbox("Load Example", ["Select an example..."] + list(examples.keys()))
        if selected_example != "Select an example...":
            st.code(examples[selected_example], language="sql")
    
    with tab4:
        # Analytics
        st.markdown("### 📈 Database Analytics")
        
        # Basic analytics
        try:
            # User statistics
            st.markdown("#### 👥 User Statistics")
            user_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN last_login IS NOT NULL THEN 1 END) as active_users,
                    COUNT(CASE WHEN is_active = 1 THEN 1 END) as enabled_users
                FROM users
            """, conn)
            
            if not user_stats.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Users", user_stats.iloc[0]['total_users'])
                with col2:
                    st.metric("Active Users", user_stats.iloc[0]['active_users'])
                with col3:
                    st.metric("Enabled Users", user_stats.iloc[0]['enabled_users'])
            
            # Document statistics
            st.markdown("#### 📄 Document Statistics")
            doc_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT user_id) as users_with_docs,
                    SUM(file_size_mb) as total_size_mb,
                    AVG(file_size_mb) as avg_size_mb
                FROM documents
            """, conn)
            
            if not doc_stats.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Documents", doc_stats.iloc[0]['total_documents'])
                with col2:
                    st.metric("Users with Docs", doc_stats.iloc[0]['users_with_docs'])
                with col3:
                    st.metric("Total Size", f"{doc_stats.iloc[0]['total_size_mb']:.1f} MB")
                with col4:
                    st.metric("Avg Size", f"{doc_stats.iloc[0]['avg_size_mb']:.1f} MB")
            
            # File type distribution
            st.markdown("#### 📁 File Type Distribution")
            file_types = pd.read_sql_query("""
                SELECT file_type, COUNT(*) as count, AVG(file_size_mb) as avg_size
                FROM documents 
                GROUP BY file_type 
                ORDER BY count DESC
            """, conn)
            
            if not file_types.empty:
                st.dataframe(file_types, use_container_width=True)
                
                # Simple chart
                try:
                    import plotly.express as px
                    fig = px.pie(file_types, values='count', names='file_type', title='Document Distribution by Type')
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("📊 Install plotly for charts: `pip install plotly`")
            
            # Recent activity
            st.markdown("#### 📅 Recent Activity")
            recent_activity = pd.read_sql_query("""
                SELECT 
                    'Document Upload' as activity,
                    name as item,
                    upload_time as timestamp
                FROM documents 
                ORDER BY upload_time DESC 
                LIMIT 10
            """, conn)
            
            if not recent_activity.empty:
                st.dataframe(recent_activity, use_container_width=True)
            else:
                st.info("📭 No recent activity found")
                
        except Exception as e:
            st.error(f"❌ Error loading analytics: {e}")
    
    with tab5:
        # Database Management
        st.markdown("### ⚙️ Database Management")
        
        # Database operations
        st.markdown("#### 🔧 Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Database Info", type="secondary"):
                st.rerun()
            
            if st.button("📊 Show Database Schema", type="secondary"):
                try:
                    schema_info = pd.read_sql_query("""
                        SELECT 
                            name as table_name,
                            sql as create_statement
                        FROM sqlite_master 
                        WHERE type='table'
                        ORDER BY name
                    """, conn)
                    
                    if not schema_info.empty:
                        st.markdown("**Database Schema:**")
                        for _, row in schema_info.iterrows():
                            with st.expander(f"📋 {row['table_name']}"):
                                st.code(row['create_statement'], language="sql")
                except Exception as e:
                    st.error(f"❌ Error loading schema: {e}")
        
        with col2:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.warning("⚠️ This will delete all documents and their chunks! Are you sure?")
                if st.button("✅ Yes, Delete All Documents", type="primary"):
                    try:
                        from core.database.connection import get_db_manager
                        db_manager = get_db_manager()
                        
                        # Delete documents and related data
                        db_manager.execute_update("DELETE FROM document_chunks")
                        db_manager.execute_update("DELETE FROM documents")
                        db_manager.execute_update("DELETE FROM vector_mappings")
                        db_manager.execute_update("DELETE FROM faiss_indices")
                        
                        # Reset auto-increment counter
                        db_manager.execute_update("DELETE FROM sqlite_sequence WHERE name='documents'")
                        
                        # Clear FAISS session state
                        from core.faiss_sync_manager import faiss_sync_manager
                        faiss_sync_manager.clear_session_state()
                        
                        st.success("✅ All documents cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing documents: {e}")
            
            if st.button("🗑️ Clear All Data", type="secondary"):
                st.warning("⚠️ This will delete ALL data including users, documents, and FAISS data! Are you sure?")
                if st.button("✅ Yes, Delete Everything", type="primary"):
                    try:
                        # Clear SQLite data using database manager
                        try:
                            from core.database.connection import get_db_manager
                            db_manager = get_db_manager()
                            
                            # Delete all data from all tables
                            db_manager.execute_update("DELETE FROM document_chunks")
                            db_manager.execute_update("DELETE FROM documents")
                            db_manager.execute_update("DELETE FROM chat_messages")
                            db_manager.execute_update("DELETE FROM chat_sessions")
                            db_manager.execute_update("DELETE FROM user_sessions")
                            db_manager.execute_update("DELETE FROM users")
                            db_manager.execute_update("DELETE FROM vector_mappings")
                            db_manager.execute_update("DELETE FROM faiss_indices")
                            
                            # Reset auto-increment counters
                            db_manager.execute_update("DELETE FROM sqlite_sequence WHERE name='documents'")
                            db_manager.execute_update("DELETE FROM sqlite_sequence WHERE name='users'")
                            db_manager.execute_update("DELETE FROM sqlite_sequence WHERE name='chat_sessions'")
                            
                            print("✅ SQLite data cleared successfully")
                            
                        except Exception as sqlite_error:
                            st.error(f"❌ Error clearing SQLite data: {sqlite_error}")
                            return
                        
                        # Clear FAISS data
                        try:
                            from core.faiss_sync_manager import faiss_sync_manager
                            user_id = st.session_state.get('user_id')
                            
                            # Clear session state first
                            faiss_sync_manager.clear_session_state()
                            
                            # Delete FAISS files
                            success = faiss_sync_manager.delete_all_data(user_id)
                            
                            if success:
                                st.success("✅ All data cleared successfully from both SQLite and FAISS!")
                            else:
                                st.warning("⚠️ SQLite cleared but FAISS deletion failed")
                        except Exception as faiss_error:
                            st.warning(f"⚠️ SQLite cleared but FAISS error: {faiss_error}")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing data: {e}")
            
            if st.button("🔄 Clear Session State", type="secondary"):
                st.warning("⚠️ This will clear all document data from memory (session state) but keep database files. Use this if documents still appear after deletion.")
                if st.button("✅ Yes, Clear Session State", type="primary"):
                    try:
                        from core.faiss_sync_manager import faiss_sync_manager
                        faiss_sync_manager.clear_session_state()
                        st.success("✅ Session state cleared! The page will refresh to show the changes.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing session state: {e}")
        
        # Database health check
        st.markdown("#### 🏥 Database Health Check")
        
        try:
            # Check for orphaned records
            orphaned_docs = pd.read_sql_query("""
                SELECT COUNT(*) as count
                FROM documents d
                LEFT JOIN users u ON d.user_id = u.id
                WHERE u.id IS NULL
            """, conn)
            
            orphaned_count = orphaned_docs.iloc[0]['count'] if not orphaned_docs.empty else 0
            
            # Check for data integrity
            integrity_issues = []
            if orphaned_count > 0:
                integrity_issues.append(f"Found {orphaned_count} documents with invalid user references")
            
            # Display health status
            if integrity_issues:
                st.error("❌ Database Health Issues Found:")
                for issue in integrity_issues:
                    st.write(f"• {issue}")
            else:
                st.success("✅ Database health check passed!")
                
        except Exception as e:
            st.error(f"❌ Error during health check: {e}")
    
    # Close database connection
    conn.close()

if __name__ == "__main__":
    main() 