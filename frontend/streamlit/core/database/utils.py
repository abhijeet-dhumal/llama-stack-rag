"""
Database utilities for RAG LlamaStack
Helper functions for database operations and management
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

def get_db_path() -> Path:
    """Get the database file path"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir / "rag_llamastack.db"

def get_connection() -> Optional[sqlite3.Connection]:
    """Get a connection to the SQLite database"""
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a query and return results as list of dictionaries"""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        return results
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []
    finally:
        conn.close()

def execute_update(query: str, params: tuple = ()) -> bool:
    """Execute an update query and return success status"""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        print(f"❌ Update error: {e}")
        return False
    finally:
        conn.close()

def get_table_schema(table_name: str) -> List[Dict[str, Any]]:
    """Get schema information for a table"""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [dict(row) for row in cursor.fetchall()]
        return columns
    except Exception as e:
        print(f"❌ Schema error: {e}")
        return []
    finally:
        conn.close()

def get_table_data(table_name: str, limit: int = 100) -> pd.DataFrame:
    """Get data from a table as pandas DataFrame"""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"❌ Data retrieval error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_database_stats() -> Dict[str, Any]:
    """Get comprehensive database statistics"""
    conn = get_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in cursor.fetchall()]
        
        stats = {
            'total_tables': len(tables),
            'tables': {},
            'total_rows': 0,
            'database_size_mb': 0
        }
        
        # Get stats for each table
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            row_count = cursor.fetchone()['count']
            stats['tables'][table] = row_count
            stats['total_rows'] += row_count
        
        # Get database file size
        db_path = get_db_path()
        if db_path.exists():
            stats['database_size_mb'] = db_path.stat().st_size / (1024 * 1024)
        
        return stats
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return {}
    finally:
        conn.close()

def backup_database(backup_path: Optional[Path] = None) -> bool:
    """Create a backup of the database"""
    if backup_path is None:
        backup_path = get_db_path().parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    
    try:
        import shutil
        shutil.copy2(get_db_path(), backup_path)
        print(f"✅ Database backed up to: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Backup error: {e}")
        return False

def export_table_to_csv(table_name: str, output_path: Path) -> bool:
    """Export a table to CSV file"""
    try:
        df = get_table_data(table_name, limit=10000)  # Limit to prevent memory issues
        df.to_csv(output_path, index=False)
        print(f"✅ Table {table_name} exported to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False

def import_csv_to_table(csv_path: Path, table_name: str, if_exists: str = 'replace') -> bool:
    """Import CSV data into a table"""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        print(f"✅ CSV imported to table {table_name}")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    finally:
        conn.close()

def check_database_integrity() -> List[str]:
    """Check database integrity and return list of issues"""
    issues = []
    
    try:
        # Check for orphaned documents
        orphaned_docs = execute_query("""
            SELECT COUNT(*) as count
            FROM documents d
            LEFT JOIN users u ON d.user_id = u.id
            WHERE u.id IS NULL
        """)
        
        if orphaned_docs and orphaned_docs[0]['count'] > 0:
            issues.append(f"Found {orphaned_docs[0]['count']} documents with invalid user references")
        
        # Check for orphaned chat messages
        orphaned_messages = execute_query("""
            SELECT COUNT(*) as count
            FROM chat_messages cm
            LEFT JOIN chat_sessions cs ON cm.chat_session_id = cs.id
            WHERE cs.id IS NULL
        """)
        
        if orphaned_messages and orphaned_messages[0]['count'] > 0:
            issues.append(f"Found {orphaned_messages[0]['count']} chat messages with invalid session references")
        
        return issues
    except Exception as e:
        issues.append(f"Integrity check error: {e}")
        return issues

def optimize_database() -> bool:
    """Optimize the database (VACUUM and ANALYZE)"""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        cursor.execute("ANALYZE")
        conn.commit()
        print("✅ Database optimized successfully")
        return True
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False
    finally:
        conn.close() 