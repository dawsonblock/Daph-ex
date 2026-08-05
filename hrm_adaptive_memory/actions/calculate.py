"""Deterministic arithmetic as an explicit action, with receipts.

Numeric derivation is a separate capability from recall, so it gets a separate
action rather than being folded into generation. The evaluator walks a
restricted AST: `eval` is never called, no names, attributes, calls,
subscripts, or comprehensions are reachable, and exponents are bounded so a
short expression cannot become a denial-of-service.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

MAX_EXPONENT = 64
MAX_ABS_OPERAND = 1e12

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_OPERATION_WORDS = (
    (re.compile(r"\bmultipl(?:ies|y|ied)\b|\btimes\b|\bproduct\b"), "*"),
    (re.compile(r"\bdivid(?:es|e|ed)\b|\bper\b"), "/"),
    (re.compile(r"\b(?:plus|sum|added|total)\b"), "+"),
    (re.compile(r"\b(?:minus|less|difference|subtract(?:ed|s)?)\b"), "-"),
)
_NUMBER = re.compile(r"(?<![\w-])[-+]?\d+(?:\.\d+)?(?![\w-])")


class UnsafeExpression(ValueError):
    """Raised when an expression falls outside the permitted arithmetic subset."""


@dataclass(frozen=True)
class CalculationReceipt:
    expression: str
    operands: tuple[str, ...]
    operation: str | None
    result: str
    source_evidence_ids: tuple[str, ...]
    verified: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpression(f"Only numeric literals are permitted, got {node.value!r}")
        if abs(node.value) > MAX_ABS_OPERAND:
            raise UnsafeExpression("Operand magnitude exceeds the permitted bound")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        handler = _BINARY.get(type(node.op))
        if handler is None:
            raise UnsafeExpression(f"Operator {type(node.op).__name__} is not permitted")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise UnsafeExpression("Exponent exceeds the permitted bound")
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise UnsafeExpression("Division by zero")
        return float(handler(left, right))
    if isinstance(node, ast.UnaryOp):
        handler = _UNARY.get(type(node.op))
        if handler is None:
            raise UnsafeExpression(f"Unary {type(node.op).__name__} is not permitted")
        return float(handler(_evaluate(node.operand)))
    raise UnsafeExpression(f"Node {type(node).__name__} is not permitted")


def safe_eval(expression: str) -> float:
    """Evaluate `+ - * / % ** ( )` over numeric literals only."""

    if len(expression) > 256:
        raise UnsafeExpression("Expression exceeds the permitted length")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise UnsafeExpression(f"Unparsable expression: {error}") from error
    return _evaluate(tree)


def _format(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(round(value, 10))


def detect_operation(text: str) -> str | None:
    lowered = text.lower()
    for pattern, symbol in _OPERATION_WORDS:
        if pattern.search(lowered):
            return symbol
    return None


def calculate_from_evidence(
    records: Sequence[Mapping[str, Any] | Any], *, operation: str | None = None,
) -> CalculationReceipt | None:
    """Derive a value from operands stated across the retrieved records.

    Returns ``None`` rather than guessing when the operands or the operation
    cannot be identified: a silent wrong number is worse than no number.
    """

    contents, ids = [], []
    for record in records:
        if isinstance(record, Mapping):
            contents.append(str(record.get("content", "")))
            ids.append(str(record.get("evidence_id", "")))
        else:
            contents.append(str(getattr(record, "content", "")))
            ids.append(str(getattr(record, "evidence_id", "")))
    joined = " ".join(contents)
    symbol = operation or detect_operation(joined)
    if symbol is None:
        return None

    operands = [value for content in contents for value in _NUMBER.findall(content)]
    if len(operands) < 2:
        return None
    # The controlled derivation shape is a single binary operation over the two
    # stated quantities; more operands mean the shape is not recognised.
    if len(operands) > 2:
        return None

    expression = f"({operands[0]}) {symbol} ({operands[1]})"
    try:
        value = safe_eval(expression)
    except UnsafeExpression as error:
        return CalculationReceipt(
            expression=expression, operands=tuple(operands), operation=symbol,
            result="", source_evidence_ids=tuple(ids), verified=False,
            rationale=f"rejected by the safe evaluator: {error}",
        )
    return CalculationReceipt(
        expression=expression,
        operands=tuple(operands),
        operation=symbol,
        result=_format(value),
        source_evidence_ids=tuple(ids),
        verified=True,
        rationale="single binary operation over the two stated operands",
    )
