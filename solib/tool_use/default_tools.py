import ast
import operator
from typing import Union


def math_eval(expr: str) -> Union[str, float]:
    """
    Evaluate a simple mathematical expression. The supported operators are * (multiply), - (subtract),
    + (add), / (divide), and ** (power). If your expression was unable to be evaluated,
    the output will instead be "Invalid expression".

    Args:
    expr: str, the expression to evaluate

    Returns:
    float, the result of the evaluation, or str "Invalid expression" if the expression is invalid
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.Pow: operator.pow
    }
    # Parse the expression into an AST
    node = ast.parse(expr, mode='eval').body

    def eval_node(node):
        if isinstance(node, ast.BinOp):
            if type(node.op) in operators:
                left = eval_node(node.left)
                right = eval_node(node.right)
                return operators[type(node.op)](left, right)
            else:
                raise ValueError(f"Unsupported operation {node.op}")
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError(f"Unsupported type {type(node)}")

    try:
        return int(eval_node(node))
    except ValueError as e:
        return "Invalid expression"
