import ast
import operator
from collections.abc import Callable

import numpy as np

# ``sympy.sympify`` evaluates arbitrary Python (including ``__import__``).
# Only these arithmetic operations on the variable ``x`` are accepted.
_BIN_OPS: dict[type[ast.operator], Callable[[object, object], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[object], object]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_UNSUPPORTED = "only arithmetic expressions in the variable x are allowed"


def get_inputs() -> tuple[str, float, float]:
    """
    Get user input for the function, lower limit, and upper limit.

    Returns:
        Tuple[str, float, float]: A tuple containing the function as a string,
        the lower limit (a), and the upper limit (b) as floats.

    Example:
        >>> from unittest.mock import patch
        >>> inputs = ['1/(1+x**2)', 1.0, -1.0]
        >>> with patch('builtins.input', side_effect=inputs):
        ...     get_inputs()
        ('1/(1+x**2)', 1.0, -1.0)
    """
    func = input("Enter function with variable as x: ")
    lower_limit = float(input("Enter lower limit: "))
    upper_limit = float(input("Enter upper limit: "))
    return func, lower_limit, upper_limit


def _validate_arithmetic(node: ast.AST) -> None:
    """Reject every AST node that is not arithmetic in the variable ``x``."""
    if isinstance(node, ast.Expression):
        _validate_arithmetic(node.body)
        return
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return
    if isinstance(node, ast.Name) and node.id == "x":
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        _validate_arithmetic(node.left)
        _validate_arithmetic(node.right)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        _validate_arithmetic(node.operand)
        return
    raise ValueError(_UNSUPPORTED)


def _eval_arithmetic(node: ast.AST, x: object) -> object:
    """Evaluate a previously validated arithmetic AST at ``x``."""
    if isinstance(node, ast.Expression):
        return _eval_arithmetic(node.body, x)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return x
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](
            _eval_arithmetic(node.left, x), _eval_arithmetic(node.right, x)
        )
    return _UNARY_OPS[type(node.op)](_eval_arithmetic(node.operand, x))


def safe_function_eval(func_str: str) -> Callable:
    """
    Build a numeric function of ``x`` from an arithmetic expression.

    Names, calls, and attribute access are rejected. This used to call
    ``sympy.sympify``, which evaluates arbitrary Python such as
    ``__import__('os').getcwd()``.

    Args:
        func_str (str): Function expression as a string.

    Returns:
        Callable: A callable function for numerical evaluation.

    Examples:
        >>> f = safe_function_eval('x**2')
        >>> f(3)
        9
        >>> f = safe_function_eval('x + x**2')
        >>> f(2)
        6
        >>> safe_function_eval("__import__('os').getcwd()")
        Traceback (most recent call last):
            ...
        ValueError: only arithmetic expressions in the variable x are allowed
    """
    try:
        tree = ast.parse(func_str, mode="eval")
    except SyntaxError as exc:
        raise ValueError(_UNSUPPORTED) from exc
    _validate_arithmetic(tree)

    def _func(value: object) -> object:
        return _eval_arithmetic(tree, value)

    return _func


def compute_table(
    func: Callable, lower_limit: float, upper_limit: float, acc: int
) -> tuple[np.ndarray, float]:
    """
    Compute the table of function values based on the limits and accuracy.

    Args:
        func (Callable): The mathematical function as a callable.
        lower_limit (float): The lower limit of the integral.
        upper_limit (float): The upper limit of the integral.
        acc (int): The number of subdivisions for accuracy.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the table
        of values and the step size (h).

    Example:
        >>> compute_table(
        ...     safe_function_eval('1/(1+x**2)'), 1, -1, 1
        ... )
        (array([0.5       , 0.69230769, 0.9       , 1.        , 0.9       ,
               0.69230769, 0.5       ]), -0.3333333333333333)
    """
    # Weddle's rule requires number of intervals as a multiple of 6 for accuracy
    n_points = acc * 6 + 1
    h = (upper_limit - lower_limit) / (n_points - 1)
    x_vals = np.linspace(lower_limit, upper_limit, n_points)
    table = func(x_vals)  # Evaluate function values at all points
    return table, h


def apply_weights(table: list[float]) -> list[float]:
    """
    Apply Weddle's rule weights to the values in the table.

    Args:
        table (List[float]): A list of computed function values.

    Returns:
        List[float]: A list of weighted values.

    Example:
        >>> apply_weights([0.0, 0.866, 1.0, 0.866, 0.0, -0.866, -1.0])
        [4.33, 1.0, 5.196, 0.0, -4.33]
    """
    add = []
    for i in range(1, len(table) - 1):
        if i % 2 == 0 and i % 3 != 0:
            add.append(table[i])
        elif i % 2 != 0 and i % 3 != 0:
            add.append(5 * table[i])
        elif i % 6 == 0:
            add.append(2 * table[i])
        elif i % 3 == 0 and i % 2 != 0:
            add.append(6 * table[i])
    return add


def compute_solution(add: list[float], table: list[float], step_size: float) -> float:
    """
    Compute the final solution using the weighted values and table.

    Args:
        add (List[float]): A list of weighted values from apply_weights.
        table (List[float]): A list of function values.
        step_size (float): The step size calculated from the limits and accuracy.

    Returns:
        float: The final computed integral solution.

    Example:
        >>> compute_solution([4.33, 6.0, 0.0, -4.33], [0.0, 0.866, 1.0, 0.866, 0.0,
        ... -0.866, -1.0], 0.5235983333333333)
        0.7853975
    """
    return 0.3 * step_size * (sum(add) + table[0] + table[-1])


if __name__ == "__main__":
    from doctest import testmod

    testmod()

    # func_str, a, b = get_inputs()
    # acc = 1
    # solution = None

    # func = safe_function_eval(func_str)
    # while acc <= 100_000:
    #     table, h = compute_table(func, a, b, acc)
    #     add = apply_weights(table)
    #     solution = compute_solution(add, table, h)
    #     acc *= 10

    # print(f"Solution: {solution}")
