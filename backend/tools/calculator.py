from __future__ import annotations

import ast
import operator
from typing import Any, Callable


ALLOWED_OPERATORS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(ALLOWED_OPERATORS[type(node.op)](left, right))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> str:
    cleaned = expression.strip()
    if not cleaned:
        return "No expression provided."
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_ast(tree)
        if result.is_integer():
            return str(int(result))
        return f"{result:.6f}".rstrip("0").rstrip(".")
    except Exception as exc:  # noqa: BLE001
        return f"Calculation error: {exc}"

