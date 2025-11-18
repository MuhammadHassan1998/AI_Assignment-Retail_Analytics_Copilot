"""DSPy signatures and modules for the retail analytics copilot."""

import json
import logging
from typing import Any, Dict, List, Optional

import dspy


class RouterSignature(dspy.Signature):
    """Route questions to appropriate handler: rag, sql, or hybrid."""
    question: str = dspy.InputField(desc="The user's question")
    route: str = dspy.OutputField(desc="One of: 'rag', 'sql', or 'hybrid'")


class NLToSQLSignature(dspy.Signature):
    """Generate SQL query from natural language question and schema."""
    question: str = dspy.InputField(desc="The user's question")
    db_schema: str = dspy.InputField(desc="Database schema information")
    retrieved_context: str = dspy.InputField(desc="Relevant document chunks (if any)")
    sql_query: str = dspy.OutputField(desc="Valid SQLite SQL query")


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from SQL results and document context."""
    question: str = dspy.InputField(desc="The original question")
    sql_results: str = dspy.InputField(desc="SQL query results as JSON string")
    document_context: str = dspy.InputField(desc="Relevant document chunks")
    format_hint: str = dspy.InputField(desc="Expected output format (e.g., 'int', 'float', '{category:str, quantity:int}')")
    final_answer: str = dspy.OutputField(desc="Final answer matching the format_hint exactly")


class Router(dspy.Module):
    """Route questions to appropriate handler."""
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str) -> str:
        result = self.classify(question=question)
        route = result.route.lower().strip()
        # Normalize to one of the three options
        if 'rag' in route or 'document' in route:
            return 'rag'
        elif 'sql' in route or 'database' in route:
            return 'sql'
        else:
            return 'hybrid'


class NLToSQL(dspy.Module):
    """Convert natural language to SQL."""
    
    def __init__(self, optimized: bool = False):
        super().__init__()
        if optimized:
            # Use optimized version with few-shot examples
            self.generate = dspy.ChainOfThought(NLToSQLSignature)
            # Add few-shot examples for better performance
            self.generate = dspy.ChainOfThought(
                NLToSQLSignature,
                examples=[
                    dspy.Example(
                        question="What are the top 3 products by revenue?",
                        db_schema="Table: Products (ProductID, ProductName, UnitPrice)\nTable: Order Details (OrderID, ProductID, UnitPrice, Quantity, Discount)",
                        retrieved_context="",
                        sql_query="SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue FROM Products p JOIN \"Order Details\" od ON p.ProductID = od.ProductID GROUP BY p.ProductID, p.ProductName ORDER BY revenue DESC LIMIT 3"
                    ).with_inputs("question", "db_schema", "retrieved_context"),
                ]
            )
        else:
            self.generate = dspy.ChainOfThought(NLToSQLSignature)
    
    def forward(self, question: str, schema: str, retrieved_context: str = "") -> str:
        result = self.generate(
            question=question,
            db_schema=schema,
            retrieved_context=retrieved_context or "No document context available."
        )
        sql = result.sql_query.strip()
        # Remove markdown code blocks if present
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()


class Synthesizer(dspy.Module):
    """Synthesize final answer from results and context."""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(self, question: str, sql_results: List[Dict[str, Any]], 
                document_context: str, format_hint: str) -> str:
        # Convert SQL results to JSON string
        sql_results_str = json.dumps(sql_results, default=str)
        
        result = self.synthesize(
            question=question,
            sql_results=sql_results_str,
            document_context=document_context or "No document context available.",
            format_hint=format_hint
        )
        
        return result.final_answer.strip()


def setup_dspy(model_name: str = "phi3.5:3.8b-mini-instruct-q4_K_M"):
    """Setup DSPy with local Ollama model."""
    try:
        # DSPy supports Ollama via OpenAI-compatible API
        # Ollama serves models at http://localhost:11434/v1
        import os

        os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
        os.environ["OPENAI_API_KEY"] = "ollama"

        # Use OpenAI class with Ollama endpoint (Ollama is OpenAI-compatible)
        lm = dspy.OpenAI(
            model=model_name,
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        )
        dspy.configure(lm=lm)
        return True
    except Exception as exc:  # pragma: no cover - connectivity issues
        logging.warning("Could not connect to Ollama: %s", exc)
        logging.warning(
            "Ensure Ollama is running, the model is pulled (`ollama pull %s`), "
            "and `ollama serve` is active.",
            model_name,
        )
        return False

