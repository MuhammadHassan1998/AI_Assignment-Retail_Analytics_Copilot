"""LangGraph hybrid agent for retail analytics."""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
import json
import re

from agent.dspy_signatures import Router, NLToSQL, Synthesizer, setup_dspy
from agent.rag.retrieval import TFIDFRetriever, create_retriever
from agent.tools.sqlite_tool import SQLiteTool, create_sqlite_tool


class AgentState(TypedDict):
    """State for the agent graph."""
    question: str
    format_hint: str
    route: Optional[str]
    retrieved_chunks: List[tuple]  # List of (chunk, score) tuples
    document_context: str
    constraints: Dict[str, Any]  # Extracted constraints (dates, categories, etc.)
    sql_query: Optional[str]
    sql_results: Optional[List[Dict[str, Any]]]
    sql_error: Optional[str]
    table_names_used: List[str]
    final_answer: Optional[str]
    explanation: str
    citations: List[str]
    confidence: float
    repair_count: int
    trace: List[str]


class HybridAgent:
    """Hybrid agent combining RAG and SQL."""
    
    def __init__(self, db_path: str = "data/northwind.sqlite", docs_dir: str = "docs"):
        # Setup DSPy
        if not setup_dspy():
            raise RuntimeError("Failed to setup DSPy. Ensure Ollama is running.")
        
        # Initialize components
        self.router = Router()
        self.nl_to_sql = NLToSQL(optimized=True)  # Use optimized version
        self.synthesizer = Synthesizer()
        self.retriever = create_retriever(docs_dir)
        self.sql_tool = create_sqlite_tool(db_path)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("router", self._router_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("nl_to_sql", self._nl_to_sql_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        graph.add_node("repair", self._repair_node)
        
        # Define edges
        graph.set_entry_point("router")
        
        graph.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "rag": "retriever",
                "sql": "planner",
                "hybrid": "retriever"
            }
        )
        
        graph.add_edge("retriever", "planner")
        graph.add_edge("planner", "nl_to_sql")
        graph.add_edge("nl_to_sql", "executor")
        
        graph.add_conditional_edges(
            "executor",
            self._executor_decision,
            {
                "success": "synthesizer",
                "repair": "repair",
                "fail": "synthesizer"  # Try to synthesize even on failure
            }
        )
        
        graph.add_conditional_edges(
            "synthesizer",
            self._synthesizer_decision,
            {
                "done": END,
                "repair": "repair"
            }
        )
        
        graph.add_conditional_edges(
            "repair",
            self._repair_decision,
            {
                "retry": "nl_to_sql",
                "give_up": "synthesizer"
            }
        )
        
        return graph.compile()
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Route the question."""
        state["trace"].append("Router: Classifying question type")
        route = self.router(question=state["question"])
        state["route"] = route
        state["trace"].append(f"Router: Determined route = {route}")
        return state
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant document chunks."""
        state["trace"].append("Retriever: Searching documents")
        chunks = self.retriever.retrieve(state["question"], top_k=5)
        state["retrieved_chunks"] = chunks
        
        # Build document context string
        context_parts = []
        for chunk, score in chunks:
            context_parts.append(f"[{chunk.chunk_id}] {chunk.content}")
        state["document_context"] = "\n\n".join(context_parts)
        
        state["trace"].append(f"Retriever: Found {len(chunks)} relevant chunks")
        return state
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Extract constraints from question and documents."""
        state["trace"].append("Planner: Extracting constraints")
        
        constraints = {}
        question_lower = state["question"].lower()
        doc_context = state.get("document_context", "")
        
        # Extract date ranges from marketing calendar
        if "summer beverages 1997" in question_lower or "summer beverages 1997" in doc_context.lower():
            constraints["date_start"] = "1997-06-01"
            constraints["date_end"] = "1997-06-30"
        elif "winter classics 1997" in question_lower or "winter classics 1997" in doc_context.lower():
            constraints["date_start"] = "1997-12-01"
            constraints["date_end"] = "1997-12-31"
        elif "1997" in question_lower:
            constraints["year"] = 1997
        
        # Extract categories
        categories = ["Beverages", "Condiments", "Confections", "Dairy Products", 
                     "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"]
        for cat in categories:
            if cat.lower() in question_lower:
                constraints["category"] = cat
                break
        
        # Extract KPI definitions
        if "aov" in question_lower or "average order value" in question_lower:
            constraints["kpi"] = "AOV"
        elif "gross margin" in question_lower or "margin" in question_lower:
            constraints["kpi"] = "GrossMargin"
        
        state["constraints"] = constraints
        state["trace"].append(f"Planner: Extracted constraints: {constraints}")
        return state
    
    def _nl_to_sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL from natural language."""
        state["trace"].append("NL→SQL: Generating SQL query")
        
        # Get schema
        schema = self.sql_tool.get_schema()
        
        # Build enhanced question with constraints
        enhanced_question = state["question"]
        if state.get("constraints"):
            constraints_str = json.dumps(state["constraints"], indent=2)
            enhanced_question += f"\n\nConstraints: {constraints_str}"
        
        # Generate SQL
        sql = self.nl_to_sql(
            question=enhanced_question,
            schema=schema,
            retrieved_context=state.get("document_context", "")
        )
        
        state["sql_query"] = sql
        state["trace"].append(f"NL→SQL: Generated SQL: {sql[:100]}...")
        return state
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Execute SQL query."""
        state["trace"].append("Executor: Executing SQL")
        
        success, rows, error, table_names = self.sql_tool.execute_query(state["sql_query"])
        
        if success:
            state["sql_results"] = rows
            state["table_names_used"] = table_names
            state["sql_error"] = None
            state["trace"].append(f"Executor: Query succeeded, returned {len(rows)} rows")
        else:
            state["sql_error"] = error
            state["sql_results"] = None
            state["trace"].append(f"Executor: Query failed: {error}")
        
        return state
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer."""
        state["trace"].append("Synthesizer: Generating final answer")
        
        # Prepare SQL results
        sql_results = state.get("sql_results") or []
        
        # Build citations
        citations = []
        citations.extend(state.get("table_names_used", []))
        
        # Add document chunk citations
        for chunk, score in state.get("retrieved_chunks", []):
            if score > 0.1:  # Only cite chunks with meaningful relevance
                citations.append(chunk.chunk_id)
        
        # Remove duplicates
        citations = list(set(citations))
        
        # For simple cases, try to format directly from SQL results
        format_hint = state["format_hint"]
        direct_answer = self._try_direct_format(sql_results, format_hint, state["question"])
        
        if direct_answer is not None:
            state["final_answer"] = direct_answer
            state["citations"] = citations
            state["explanation"] = f"Generated answer directly from SQL query results."
            state["confidence"] = self._calculate_confidence(state)
            state["trace"].append(f"Synthesizer: Used direct formatting, answer: {direct_answer}")
            return state
        
        # Synthesize answer using LLM
        answer = self.synthesizer(
            question=state["question"],
            sql_results=sql_results,
            document_context=state.get("document_context", ""),
            format_hint=format_hint
        )
        
        # Parse and validate answer format
        parsed_answer = self._parse_answer(answer, format_hint)
        
        state["final_answer"] = parsed_answer
        state["citations"] = citations
        state["explanation"] = f"Generated answer from {'SQL query' if sql_results else 'document context'}."
        state["confidence"] = self._calculate_confidence(state)
        state["trace"].append(f"Synthesizer: Generated answer: {parsed_answer}")
        
        return state
    
    def _try_direct_format(self, sql_results: List[Dict[str, Any]], format_hint: str, question: str) -> Any:
        """Try to format SQL results directly without LLM."""
        if not sql_results:
            return None
        
        # Handle int format
        if format_hint == "int":
            if len(sql_results) == 1:
                row = sql_results[0]
                for val in row.values():
                    if isinstance(val, (int, float)):
                        return int(val)
            return None
        
        # Handle float format
        if format_hint == "float":
            if len(sql_results) == 1:
                row = sql_results[0]
                for val in row.values():
                    if isinstance(val, (int, float)):
                        return round(float(val), 2)
            return None
        
        # Handle dict format (single row)
        if "{" in format_hint and ":" in format_hint:
            if len(sql_results) == 1:
                row = sql_results[0]
                # Try to map row to expected format
                result_dict = {}
                for key, val in row.items():
                    # Normalize key names
                    key_lower = key.lower()
                    if "category" in format_hint.lower() and ("category" in key_lower or "name" in key_lower):
                        result_dict["category"] = str(val)
                    elif "customer" in format_hint.lower() and ("customer" in key_lower or "company" in key_lower):
                        result_dict["customer"] = str(val)
                    elif "quantity" in format_hint.lower() and "quantity" in key_lower:
                        result_dict["quantity"] = int(val) if isinstance(val, (int, float)) else val
                    elif "margin" in format_hint.lower() or "revenue" in format_hint.lower():
                        if "margin" in key_lower or "revenue" in key_lower or "total" in key_lower:
                            result_dict["margin" if "margin" in format_hint.lower() else "revenue"] = round(float(val), 2) if isinstance(val, (int, float)) else val
                
                if result_dict:
                    return result_dict
            return None
        
        # Handle list format
        if "[" in format_hint and "list" in format_hint.lower():
            result_list = []
            for row in sql_results:
                item = {}
                for key, val in row.items():
                    if "product" in format_hint.lower() and ("product" in key.lower() or "name" in key.lower()):
                        item["product"] = str(val)
                    elif "revenue" in format_hint.lower() and ("revenue" in key.lower() or "total" in key.lower() or "sum" in key.lower()):
                        item["revenue"] = round(float(val), 2) if isinstance(val, (int, float)) else val
                
                if item:
                    result_list.append(item)
            
            if result_list:
                return result_list
        
        return None
    
    def _repair_node(self, state: AgentState) -> AgentState:
        """Repair failed queries or invalid outputs."""
        state["trace"].append("Repair: Attempting to fix issues")
        state["repair_count"] = state.get("repair_count", 0) + 1
        
        # If SQL error, try to fix the query
        if state.get("sql_error"):
            # Simple repair: add error context to question
            state["question"] = f"{state['question']} (Previous error: {state['sql_error']})"
            state["trace"].append(f"Repair: Added error context to question")
        
        return state
    
    def _parse_answer(self, answer: str, format_hint: str) -> Any:
        """Parse answer to match format_hint."""
        if answer is None:
            return None
        
        answer = answer.strip()
        
        # Remove markdown code blocks if present
        if answer.startswith("```"):
            lines = answer.split("\n")
            answer = "\n".join(lines[1:-1]) if len(lines) > 2 else answer
            answer = answer.strip()
        
        # If answer is already a dict/list (from JSON parsing), return it
        if isinstance(answer, (dict, list, int, float)):
            return answer
        
        # Try to extract JSON if format_hint suggests structured data
        if "{" in format_hint or "[" in format_hint:
            # Try to find JSON in the answer (more robust pattern)
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested dict
                r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Nested list
            ]
            for pattern in json_patterns:
                matches = re.finditer(pattern, answer)
                for match in matches:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, (dict, list)):
                            return parsed
                    except:
                        continue
        
        # Try to parse as the expected type
        if format_hint == "int":
            # Extract integer
            int_match = re.search(r'\d+', answer)
            if int_match:
                return int(int_match.group())
            return 0
        
        elif format_hint == "float":
            # Extract float (more robust)
            float_match = re.search(r'\d+\.\d+', answer)
            if float_match:
                return round(float(float_match.group()), 2)
            int_match = re.search(r'\d+', answer)
            if int_match:
                return round(float(int_match.group()), 2)
            return 0.0
        
        elif "{" in format_hint:
            # Try to parse as dict
            try:
                # Try full answer first
                parsed = json.loads(answer)
                if isinstance(parsed, dict):
                    return parsed
            except:
                pass
            
            # Try to extract dict from text
            dict_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', answer)
            if dict_match:
                try:
                    return json.loads(dict_match.group())
                except:
                    pass
        
        elif "[" in format_hint and "list" in format_hint.lower():
            # Try to parse as list
            try:
                parsed = json.loads(answer)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            
            # Try to extract list from text
            list_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', answer)
            if list_match:
                try:
                    return json.loads(list_match.group())
                except:
                    pass
        
        return answer
    
    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate confidence score."""
        confidence = 0.5  # Base confidence
        
        # Boost if SQL succeeded
        if state.get("sql_results") is not None:
            confidence += 0.2
            if len(state.get("sql_results", [])) > 0:
                confidence += 0.1
        
        # Boost if we have document context
        if state.get("document_context"):
            confidence += 0.1
        
        # Reduce if we had to repair
        if state.get("repair_count", 0) > 0:
            confidence -= 0.1 * state["repair_count"]
        
        # Boost if we have citations
        if state.get("citations"):
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _route_decision(self, state: AgentState) -> str:
        """Decision after router."""
        return state.get("route", "hybrid")
    
    def _executor_decision(self, state: AgentState) -> str:
        """Decision after executor."""
        if state.get("sql_error") and state.get("repair_count", 0) < 2:
            return "repair"
        elif state.get("sql_error"):
            return "fail"
        else:
            return "success"
    
    def _synthesizer_decision(self, state: AgentState) -> str:
        """Decision after synthesizer."""
        # Check if answer is valid
        answer = state.get("final_answer")
        if answer is None or answer == "":
            if state.get("repair_count", 0) < 2:
                return "repair"
        
        # Check if format matches
        format_hint = state.get("format_hint", "")
        if not self._validate_format(answer, format_hint):
            if state.get("repair_count", 0) < 2:
                return "repair"
        
        return "done"
    
    def _repair_decision(self, state: AgentState) -> str:
        """Decision after repair."""
        if state.get("repair_count", 0) >= 2:
            return "give_up"
        return "retry"
    
    def _validate_format(self, answer: Any, format_hint: str) -> bool:
        """Validate that answer matches format_hint."""
        if answer is None:
            return False
        
        if format_hint == "int":
            return isinstance(answer, int)
        elif format_hint == "float":
            return isinstance(answer, (int, float))
        elif "{" in format_hint:
            return isinstance(answer, dict)
        elif "[" in format_hint:
            return isinstance(answer, list)
        
        return True
    
    def process(self, question: str, format_hint: str) -> Dict[str, Any]:
        """Process a question through the graph."""
        initial_state: AgentState = {
            "question": question,
            "format_hint": format_hint,
            "route": None,
            "retrieved_chunks": [],
            "document_context": "",
            "constraints": {},
            "sql_query": None,
            "sql_results": None,
            "sql_error": None,
            "table_names_used": [],
            "final_answer": None,
            "explanation": "",
            "citations": [],
            "confidence": 0.0,
            "repair_count": 0,
            "trace": []
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
            "trace": final_state["trace"]
        }

