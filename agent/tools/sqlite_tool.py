"""SQLite tool for database access and schema introspection."""

import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class SQLiteTool:
    """Tool for executing SQL queries and introspecting schema."""
    
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    def get_schema(self) -> str:
        """Get the database schema as a string."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_parts = []
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info(`{table}`)")
            columns = cursor.fetchall()
            
            schema_parts.append(f"\nTable: {table}")
            schema_parts.append("Columns:")
            for col in columns:
                col_name, col_type = col[1], col[2]
                schema_parts.append(f"  - {col_name} ({col_type})")
            
            # Get sample data count
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            count = cursor.fetchone()[0]
            schema_parts.append(f"  Row count: {count}")
        
        # Get views
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
        views = [row[0] for row in cursor.fetchall()]
        if views:
            schema_parts.append("\nViews:")
            for view in views:
                schema_parts.append(f"  - {view}")
        
        conn.close()
        return "\n".join(schema_parts)
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    
    def execute_query(self, query: str) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str], List[str]]:
        """
        Execute a SQL query.
        
        Returns:
            (success, rows, error_message, table_names_used)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            result_rows = [dict(row) for row in rows]
            
            # Extract table names from query (simple heuristic)
            table_names = self._extract_table_names(query)
            
            conn.close()
            return True, result_rows, None, table_names
            
        except sqlite3.Error as e:
            error_msg = str(e)
            conn.close()
            return False, None, error_msg, []
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from SQL query (simple heuristic)."""
        # Get all known tables
        all_tables = self.get_table_names()
        
        # Also check for views
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
        views = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        all_names = all_tables + views
        query_upper = query.upper()
        
        found_tables = []
        for name in all_names:
            # Check if table name appears in query (case insensitive)
            if name.lower() in query.lower() or f'`{name}`' in query or f'"{name}"' in query:
                found_tables.append(name)
        
        return found_tables
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample data from a table."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {limit}")
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            conn.close()
            return result
        except sqlite3.Error:
            conn.close()
            return []


def create_sqlite_tool(db_path: str = "data/northwind.sqlite") -> SQLiteTool:
    """Factory function to create a SQLite tool."""
    return SQLiteTool(db_path)

