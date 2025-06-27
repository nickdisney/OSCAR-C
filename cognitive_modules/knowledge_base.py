# --- START OF FILE knowledge_base.py ---

# --- START OF CORRECTED knowledge_base.py (MRO Fix) ---

import asyncio
import sqlite3
import json
import logging
from pathlib import Path
from typing import Set, Optional, List, Dict, Any, Tuple

# --- Use standard relative imports ---
# Assumes this file is in cognitive_modules/ and protocols.py is one level up
try:
    # Import protocols and specific data types needed
    from ..protocols import StateQueryable # Import only the most specific protocol needed
    from ..models.datatypes import Predicate
    # Import base CognitiveComponent ONLY for type checking IF NEEDED, but don't inherit from it directly here
    from ..protocols import CognitiveComponent
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("KnowledgeBase: Relative imports failed, relying on globally defined types.")
    if 'StateQueryable' not in globals(): raise ImportError("StateQueryable not found via relative import or globally")
    if 'Predicate' not in globals(): raise ImportError("Predicate not found via relative import or globally")
    StateQueryable = globals().get('StateQueryable')
    Predicate = globals().get('Predicate')
    # CognitiveComponent is implicitly included via StateQueryable inheritance

logger_knowledge_base = logging.getLogger(__name__)

# --- Inherit ONLY from StateQueryable ---
class KnowledgeBase(StateQueryable):
    """SQLite-backed predicate knowledge base for OSCAR-C."""

    def __init__(self):
        self.db_path: Optional[Path] = None
        self._lock = asyncio.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._controller: Optional[Any] = None # Store controller reference if needed
        # self._config: Dict[str, Any] = {} # Keep if KB has other specific params, not used for db_path now

    # --- Rest of the methods (initialize, assert_fact, query, check, query_state, etc.) remain the same ---
    # --- They already correctly implement methods from both protocols via StateQueryable ---

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize database connection and schema using global config for paths."""
        self._controller = controller 
        # self._config = config.get("knowledge_base", {}) # KB-specific config might still be useful for other params

        # --- Path Configuration using agent_root_path ---
        if not (controller and hasattr(controller, 'agent_root_path')):
            logger_knowledge_base.error("KnowledgeBase: Controller or agent_root_path not available. Cannot determine DB path.")
            return False
        
        agent_root = controller.agent_root_path
        agent_data_paths_config = config.get("agent_data_paths", {})
        
        # Get the relative path string from the centralized [agent_data_paths] section
        # Default to "data/oscar_c_kb.db" if not found, consistent with new config.toml
        kb_db_relative_path_str = agent_data_paths_config.get("kb_db_path", "data/oscar_c_kb.db")

        if not kb_db_relative_path_str:
            logger_knowledge_base.error("KnowledgeBase: kb_db_path not specified in [agent_data_paths] configuration.")
            return False

        # Construct the absolute path
        # If kb_db_relative_path_str happens to be absolute, Path() will handle it correctly.
        self.db_path = (Path(agent_root) / kb_db_relative_path_str).resolve()
        logger_knowledge_base.info(f"KnowledgeBase initializing with DB path: {self.db_path}")
        # --- End Path Configuration ---

        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._cursor = self._connection.cursor()

            async with self._lock:
                self._cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predicates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        args TEXT NOT NULL,
                        value BOOLEAN NOT NULL,
                        timestamp REAL NOT NULL,
                        UNIQUE(name, args, value)
                    )
                """)
                self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_predicate_name ON predicates(name)")
                self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_predicate_timestamp ON predicates(timestamp)")
                self._connection.commit()

            logger_knowledge_base.info("KnowledgeBase initialized successfully.")
            return True

        except sqlite3.Error as e:
            logger_knowledge_base.exception(f"KnowledgeBase: SQLite error during initialization: {e}")
            if self._cursor: self._cursor.close(); self._cursor = None
            if self._connection: self._connection.close(); self._connection = None
            return False
        except Exception as e:
            logger_knowledge_base.exception(f"KnowledgeBase: Unexpected error during initialization: {e}")
            if self._cursor: self._cursor.close(); self._cursor = None
            if self._connection: self._connection.close(); self._connection = None
            return False

    async def assert_fact(self, predicate: 'Predicate'): # Use quotes for forward ref
        """Assert a fact (predicate), replacing the opposite if it exists."""
        if not self._connection or not self._cursor:
            logger_knowledge_base.error("KnowledgeBase: Cannot assert fact, database not initialized.")
            return
        # Check if Predicate class is available (could be via import or global)
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass:
            logger_knowledge_base.error("KnowledgeBase: Predicate class not available for assert_fact.")
            return
        # Ensure the input is actually a Predicate instance
        if not isinstance(predicate, _PredicateClass):
            logger_knowledge_base.error(f"KnowledgeBase: assert_fact received invalid type: {type(predicate)}")
            return

        async with self._lock:
            try:
                # Serialize args tuple to JSON string for storage
                args_json = json.dumps(predicate.args, sort_keys=True) # Sort keys for consistency

                # Remove the opposite fact (same name, same args, different value)
                self._cursor.execute("""
                    DELETE FROM predicates
                    WHERE name = ? AND args = ? AND value = ?
                """, (predicate.name, args_json, not predicate.value))
                deleted_count = self._cursor.rowcount
                if deleted_count > 0:
                     logger_knowledge_base.debug(f"Removed opposite fact for: {predicate.name}{predicate.args}")

                # Insert or replace the new fact
                # Using INSERT OR REPLACE based on UNIQUE constraint (name, args, value)
                self._cursor.execute("""
                    INSERT OR REPLACE INTO predicates (name, args, value, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (predicate.name, args_json, predicate.value, predicate.timestamp))

                self._connection.commit()
                logger_knowledge_base.debug(f"Asserted fact: {predicate.name}{predicate.args}={predicate.value}")

            except json.JSONDecodeError as e:
                 logger_knowledge_base.error(f"KnowledgeBase: Failed to serialize args for predicate {predicate.name}: {e}")
            except sqlite3.Error as e:
                logger_knowledge_base.exception(f"KnowledgeBase: SQLite error during assert_fact for {predicate.name}: {e}")
                # Attempt to rollback transaction on error
                try: self._connection.rollback()
                except Exception as rb_e: logger_knowledge_base.error(f"Rollback failed: {rb_e}")
            except Exception as e:
                 logger_knowledge_base.exception(f"KnowledgeBase: Unexpected error during assert_fact: {e}")


    async def query(self, name: str, args: Optional[Tuple[Any, ...]] = None,
                    value: Optional[bool] = True) -> List['Predicate']: # Use quotes
        """Query predicates with optional exact argument matching."""
        if not self._connection or not self._cursor:
            logger_knowledge_base.error("KnowledgeBase: Cannot query, database not initialized.")
            return []
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass:
            logger_knowledge_base.error("KnowledgeBase: Predicate class not available for query.")
            return []

        results: List[Predicate] = []
        async with self._lock:
            try:
                query_sql = "SELECT name, args, value, timestamp FROM predicates WHERE name = ? AND value = ?"
                params: List[Any] = [name, value]

                if args is not None:
                    # Requires exact match of arguments
                    try:
                        args_json = json.dumps(args, sort_keys=True)
                        query_sql += " AND args = ?"
                        params.append(args_json)
                    except TypeError as json_err:
                         logger_knowledge_base.error(f"Cannot serialize args for query {name}: {args} - {json_err}")
                         return [] # Cannot perform query with non-serializable args

                self._cursor.execute(query_sql, tuple(params)) # Params must be a tuple

                for row in self._cursor.fetchall():
                    try:
                         # Deserialize args JSON string back to tuple
                         # Ensure args_tuple is always a tuple, even if JSON was a single item
                         deserialized_args = json.loads(row["args"])
                         # Handle simple values vs lists/tuples
                         if isinstance(deserialized_args, list):
                              args_tuple = tuple(deserialized_args)
                         elif isinstance(deserialized_args, tuple):
                              args_tuple = deserialized_args # Already a tuple
                         else: # Assume single value
                              args_tuple = (deserialized_args,)


                         results.append(_PredicateClass(
                            name=row["name"],
                            args=args_tuple,
                            value=bool(row["value"]), # Ensure boolean type
                            timestamp=row["timestamp"]
                        ))
                    except json.JSONDecodeError as e:
                         logger_knowledge_base.error(f"Failed to decode args JSON '{row['args']}' for predicate {row['name']}: {e}")
                    except Exception as e_row:
                         logger_knowledge_base.error(f"Error processing row for predicate {row['name']}: {e_row}")

            except sqlite3.Error as e:
                 logger_knowledge_base.exception(f"KnowledgeBase: SQLite error during query for {name}: {e}")
            except Exception as e:
                 logger_knowledge_base.exception(f"KnowledgeBase: Unexpected error during query: {e}")

        logger_knowledge_base.debug(f"Query '{name}' (Args: {args}, Value: {value}) returned {len(results)} results.")
        return results

    async def check(self, predicate: 'Predicate') -> bool: # Use quotes
        """Check if a specific predicate exists in the knowledge base."""
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass or not isinstance(predicate, _PredicateClass):
             logger_knowledge_base.error(f"KB Check failed: Invalid predicate type {type(predicate)}")
             return False
        results = await self.query(predicate.name, predicate.args, predicate.value)
        return len(results) > 0

    # --- StateQueryable Implementation ---

    async def query_state(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Handle structured queries about the KB's state."""
        if not self._connection or not self._cursor:
            return {"error": "KnowledgeBase not initialized"}

        # Need Predicate class defined for 'all_facts' query
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass:
            return {"error": "Predicate class not available for query_state."}

        async with self._lock:
            try:
                if "predicate_count" in query:
                    self._cursor.execute("SELECT COUNT(*) FROM predicates")
                    count = self._cursor.fetchone()[0]
                    return {"predicate_count": count}

                if "recent_facts" in query:
                    n = query.get("n", 10) # Allow specifying number of recent facts
                    if not isinstance(n, int) or n <= 0: n = 10 # Default if invalid
                    self._cursor.execute("""
                        SELECT name, args, value, timestamp
                        FROM predicates
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (n,))
                    recent = []
                    for row in self._cursor.fetchall():
                        try:
                            deserialized_args = json.loads(row["args"])
                            # Handle simple values vs lists/tuples
                            if isinstance(deserialized_args, list): args_tuple = tuple(deserialized_args)
                            elif isinstance(deserialized_args, tuple): args_tuple = deserialized_args
                            else: args_tuple = (deserialized_args,)

                            recent.append({
                                "name": row["name"],
                                "args": args_tuple,
                                "value": bool(row["value"]),
                                "timestamp": row["timestamp"]
                            })
                        except Exception as e_row:
                            logger_knowledge_base.warning(f"Error processing recent fact row: {e_row}")
                    return {"recent_facts": recent}


                if "name_like" in query:
                     pattern = query.get("name_like", "%") # Use % if pattern missing
                     if not isinstance(pattern, str): pattern = "%"
                     # Basic sanitization: allow wildcard but maybe not complex patterns
                     pattern = pattern.replace('*', '%').replace('?', '_')
                     self._cursor.execute("SELECT COUNT(*) FROM predicates WHERE name LIKE ?", (f"%{pattern}%",))
                     count = self._cursor.fetchone()[0]
                     return {"name_like_count": count, "pattern": query["name_like"]} # Return original pattern

                if "all_facts" in query:
                    # Consider adding LIMIT if KB could be huge
                    limit = query.get("limit")
                    sql = "SELECT name, args, value, timestamp FROM predicates WHERE value = 1" # Fetch only true facts for state?
                    params = ()
                    if isinstance(limit, int) and limit > 0:
                         sql += " LIMIT ?"
                         params = (limit,)

                    self._cursor.execute(sql, params)
                    all_facts = []
                    for row in self._cursor.fetchall():
                        try:
                            deserialized_args = json.loads(row["args"])
                            # Handle simple values vs lists/tuples
                            if isinstance(deserialized_args, list): args_tuple = tuple(deserialized_args)
                            elif isinstance(deserialized_args, tuple): args_tuple = deserialized_args
                            else: args_tuple = (deserialized_args,)

                            all_facts.append(_PredicateClass(
                                name=row["name"],
                                args=args_tuple,
                                value=bool(row["value"]),
                                timestamp=row["timestamp"]
                            ))
                        except Exception as e_row:
                            logger_knowledge_base.warning(f"Error processing fact row for 'all_facts': {e_row}")
                    return {"all_facts": all_facts}


                return {"error": "Unsupported KB query type"}

            except sqlite3.Error as e:
                 logger_knowledge_base.exception(f"KnowledgeBase: SQLite error during query_state: {e}")
                 return {"error": f"SQLite error: {e}"}
            except Exception as e:
                 logger_knowledge_base.exception(f"KnowledgeBase: Unexpected error during query_state: {e}")
                 return {"error": f"Unexpected error: {e}"}


    # --- CognitiveComponent Implementation ---

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """The KB primarily reacts to assert_fact calls, process is likely minimal."""
        return None # KB doesn't actively produce output in the cycle directly

    async def reset(self) -> None:
        """Reset the knowledge base by clearing the table."""
        if not self._connection or not self._cursor:
            logger_knowledge_base.error("KnowledgeBase: Cannot reset, database not initialized.")
            return
        logger_knowledge_base.warning("KnowledgeBase: Resetting - Clearing all predicates!")
        async with self._lock:
            try:
                self._cursor.execute("DELETE FROM predicates")
                self._connection.commit()
                logger_knowledge_base.info("KnowledgeBase reset complete.")
            except sqlite3.Error as e:
                logger_knowledge_base.exception(f"KnowledgeBase: SQLite error during reset: {e}")
                try: self._connection.rollback()
                except Exception as rb_e: logger_knowledge_base.error(f"Rollback failed: {rb_e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get KB status summary, including DB size and fact count."""
        # Same logic as before...
        status = { "component": "KnowledgeBase", "status": "uninitialized", "fact_count": 0, "db_size_mb": 0.0, "oldest_fact_ts": None, "newest_fact_ts": None }
        if not self._connection or not self._cursor or not self.db_path: return status
        async with self._lock:
            try:
                 status["status"] = "operational"
                 self._cursor.execute("SELECT COUNT(*) FROM predicates"); count = self._cursor.fetchone()[0]; status["fact_count"] = count
                 db_size_bytes = -1
                 try:
                     if self.db_path.exists(): status["db_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 3)
                     else: status["db_size_mb"] = 0.0
                 except Exception as e_stat: logger_knowledge_base.warning(f"Could not get DB file size: {e_stat}"); status["db_size_mb"] = -1.0
                 if count > 0: self._cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM predicates"); min_ts, max_ts = self._cursor.fetchone(); status["oldest_fact_ts"] = min_ts; status["newest_fact_ts"] = max_ts
            except sqlite3.Error as e: logger_knowledge_base.exception(f"KB SQLite error get_status: {e}"); status["status"] = "error"; status["error_message"] = str(e)
            except Exception as e: logger_knowledge_base.exception(f"KB Unexpected error get_status: {e}"); status["status"] = "error"; status["error_message"] = str(e)
        return status


    async def shutdown(self) -> None:
        """Close the database connection."""
        logger_knowledge_base.info("KnowledgeBase shutting down...")
        async with self._lock:
            if self._connection:
                try:
                    self._connection.commit() # Ensure any final changes are saved
                    self._connection.close()
                    logger_knowledge_base.info("KnowledgeBase database connection closed.")
                except sqlite3.Error as e:
                    logger_knowledge_base.exception(f"KnowledgeBase: Error closing database connection: {e}")
            self._connection = None
            self._cursor = None

# --- END OF CORRECTED knowledge_base.py (MRO Fix) ---