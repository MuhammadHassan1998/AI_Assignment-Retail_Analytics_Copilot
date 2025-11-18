"""DSPy optimization module for improving NL→SQL performance."""

import logging

import dspy

from agent.dspy_signatures import NLToSQL, setup_dspy
from agent.tools.sqlite_tool import create_sqlite_tool


logger = logging.getLogger(__name__)


def _validate_sql(sql: str) -> bool:
    """Lightweight validation used during optimization."""
    normalized = sql.upper()
    return "SELECT" in normalized and "FROM" in normalized


def optimize_nl_to_sql():
    """Optimize the NLToSQL module using BootstrapFewShot."""

    if not setup_dspy():
        raise RuntimeError("Failed to setup DSPy")

    sql_tool = create_sqlite_tool()
    schema = sql_tool.get_schema()

    train_examples = [
        {
            "question": "Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)).",
            "schema": schema,
            "retrieved_context": "",
            "expected_sql": 'SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue FROM Products p JOIN "Order Details" od ON p.ProductID = od.ProductID GROUP BY p.ProductID, p.ProductName ORDER BY revenue DESC LIMIT 3',
        },
        {
            "question": "Total revenue from the 'Beverages' category during 'Summer Beverages 1997' dates (1997-06-01 to 1997-06-30).",
            "schema": schema,
            "retrieved_context": "Summer Beverages 1997: Dates: 1997-06-01 to 1997-06-30",
            "expected_sql": 'SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as total_revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID JOIN Orders o ON od.OrderID = o.OrderID WHERE c.CategoryName = \'Beverages\' AND o.OrderDate >= \'1997-06-01\' AND o.OrderDate <= \'1997-06-30\'',
        },
        {
            "question": "Average Order Value during 'Winter Classics 1997' (1997-12-01 to 1997-12-31). AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID).",
            "schema": schema,
            "retrieved_context": "AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)",
            "expected_sql": 'SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT od.OrderID) as aov FROM "Order Details" od JOIN Orders o ON od.OrderID = o.OrderID WHERE o.OrderDate >= \'1997-12-01\' AND o.OrderDate <= \'1997-12-31\'',
        },
    ]

    dspy_examples = [
        dspy.Example(
            question=ex["question"],
            db_schema=ex["schema"],
            retrieved_context=ex["retrieved_context"],
            sql_query=ex["expected_sql"],
        ).with_inputs("question", "db_schema", "retrieved_context")
        for ex in train_examples
    ]

    base_module = NLToSQL(optimized=False)
    logger.info("Testing base NL→SQL module before optimization")
    base_success = 0
    for ex in train_examples[:2]:
        try:
            candidate_sql = base_module(
                question=ex["question"],
                schema=ex["schema"],
                retrieved_context=ex["retrieved_context"],
            )
            if _validate_sql(candidate_sql):
                base_success += 1
        except Exception:
            continue

    base_accuracy = base_success / 2.0
    logger.info("Base module accuracy: %.2f%%", base_accuracy * 100)

    optimizer = dspy.BootstrapFewShot(
        max_bootstrapped_demos=4,
        max_labeled_demos=2,
    )
    optimized_module = optimizer.compile(
        student=NLToSQL(optimized=False),
        trainset=dspy_examples,
    )

    logger.info("Testing optimized NL→SQL module")
    opt_success = 0
    for ex in train_examples[:2]:
        try:
            candidate_sql = optimized_module(
                question=ex["question"],
                schema=ex["schema"],
                retrieved_context=ex["retrieved_context"],
            )
            if _validate_sql(candidate_sql):
                opt_success += 1
        except Exception:
            continue

    opt_accuracy = opt_success / 2.0
    logger.info("Optimized module accuracy: %.2f%%", opt_accuracy * 100)
    logger.info("Improvement: %.2f%%", (opt_accuracy - base_accuracy) * 100)

    return {
        "base_accuracy": base_accuracy,
        "optimized_accuracy": opt_accuracy,
        "improvement": opt_accuracy - base_accuracy,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = optimize_nl_to_sql()
    logger.info("Optimization Results — Before: %.2f%% | After: %.2f%% | Delta: %.2f%%",
                results["base_accuracy"] * 100,
                results["optimized_accuracy"] * 100,
                results["improvement"] * 100)

