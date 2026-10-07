"""
Calculates the First Fundamental Form of a parametric surface.

The first fundamental form allows the measurement of lengths, angles,
and areas on a surface defined by parametric equations r(u, v).

Reference:
- https://en.wikipedia.org/wiki/First_fundamental_form

Author: Chahat Sandhu
GitHub: https://github.com/singhc7
"""

import ast

import sympy as sp

_ALLOWED_FUNCS = {
    "acos": sp.acos,
    "asin": sp.asin,
    "atan": sp.atan,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}
_BIN_OPS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
}
_UNSUPPORTED = "only arithmetic expressions in u and v are allowed"


def first_fundamental_form(
    x_expr: str, y_expr: str, z_expr: str
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Calculate the First Fundamental Form coefficients (E, F, G) for a surface
    defined by parametric equations x(u,v), y(u,v), z(u,v).

    Args:
        x_expr: A string representing the x component in terms of u and v.
        y_expr: A string representing the y component in terms of u and v.
        z_expr: A string representing the z component in terms of u and v.

    Returns:
        A tuple containing the sympy expressions for E, F, and G.

    Examples:
        >>> # Example 1: A simple plane r(u, v) = <u, v, 0>
        >>> E, F, G = first_fundamental_form("u", "v", "0")
        >>> print(f"E: {E}, F: {F}, G: {G}")
        E: 1, F: 0, G: 1

        >>> # Example 2: A paraboloid r(u, v) = <u, v, u**2 + v**2>
        >>> E, F, G = first_fundamental_form("u", "v", "u**2 + v**2")
        >>> print(f"E: {E}, F: {F}, G: {G}")
        E: 4*u**2 + 1, F: 4*u*v, G: 4*v**2 + 1

        >>> # Example 3: A cylinder r(u, v) = <cos(u), sin(u), v>
        >>> E, F, G = first_fundamental_form("cos(u)", "sin(u)", "v")
        >>> print(f"E: {sp.simplify(E)}, F: {F}, G: {G}")
        E: 1, F: 0, G: 1

        >>> first_fundamental_form("__import__('os').getcwd()", "v", "0")
        Traceback (most recent call last):
            ...
        ValueError: only arithmetic expressions in u and v are allowed
    """
    # Define the mathematical symbols
    u, v = sp.symbols("u v")

    # Parse with a restricted AST. sympy.sympify evaluates arbitrary Python.
    x = _parse_component(x_expr, {"u": u, "v": v})
    y = _parse_component(y_expr, {"u": u, "v": v})
    z = _parse_component(z_expr, {"u": u, "v": v})

    # Define the position vector r
    r = sp.Matrix([x, y, z])

    # Calculate partial derivatives (tangent vectors)
    r_u = sp.diff(r, u)
    r_v = sp.diff(r, v)

    # Compute the coefficients E, F, G using the dot product
    # We use simplify to combine trigonometric terms where possible
    e = sp.simplify(r_u.dot(r_u))
    f = sp.simplify(r_u.dot(r_v))
    g = sp.simplify(r_v.dot(r_v))

    return e, f, g


def _parse_component(expr: str, symbols: dict[str, sp.Expr]) -> sp.Expr:
    """Turn ``expr`` into a SymPy expression without evaluating Python."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(_UNSUPPORTED) from exc
    return _eval_component(tree, symbols)


def _eval_component(node: ast.AST, symbols: dict[str, sp.Expr]) -> sp.Expr:
    """Evaluate a restricted arithmetic AST into a SymPy expression."""
    if isinstance(node, ast.Expression):
        return _eval_component(node.body, symbols)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        if isinstance(node.value, int):
            return sp.Integer(node.value)
        return sp.Float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise ValueError(_UNSUPPORTED)
        return symbols[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _eval_component(node.left, symbols),
            _eval_component(node.right, symbols),
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_component(node.operand, symbols)
        return value if isinstance(node.op, ast.UAdd) else -value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _ALLOWED_FUNCS
        and not node.keywords
    ):
        args = [_eval_component(arg, symbols) for arg in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError(_UNSUPPORTED)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
