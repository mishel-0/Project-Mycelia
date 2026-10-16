"""Restricted evaluator for documentation examples and synthesized programs.

Never uses eval/exec. Expressions are parsed to an AST and walked by hand;
only literals, the variable `x`, unary minus, subscripts by literal ints and
calls of allowlisted *pure* str methods (or sep.join) are permitted. Integer
arguments and result sizes are capped. Anything else raises SandboxError.
"""
from __future__ import annotations
import ast

PURE_STR_METHODS = frozenset('''capitalize casefold center count endswith find isalnum isalpha isdigit islower
isspace istitle isupper join ljust lower lstrip partition removeprefix removesuffix replace rfind rjust rpartition
rsplit rstrip split splitlines startswith strip swapcase title upper zfill'''.split())
MAX_INT, MAX_LEN, MAX_DIGITS = 1000, 10000, 100
SAFE_BUILTINS = {'len': len, 'sorted': sorted, 'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
                 'int': int, 'float': float, 'str': str}
SAFE_MODULE_FUNCTIONS = {
    'math': frozenset('sqrt floor ceil gcd lcm factorial isqrt prod fsum fabs trunc log10 log2 exp comb perm'.split()),
    'statistics': frozenset('mean fmean median median_low median_high median_grouped mode multimode pstdev stdev '
                            'pvariance variance geometric_mean harmonic_mean'.split()),
}


class SandboxError(Exception):
    pass


def _check_size(v):
    if isinstance(v, (str, list, tuple)) and len(v) > MAX_LEN:
        raise SandboxError('result too large')
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 10 ** MAX_DIGITS:
        raise SandboxError('number too large')
    return v


def _args(node, env, allowed):
    args = [_eval(a, env, allowed) for a in node.args]
    kwargs = {k.arg: _eval(k.value, env, allowed) for k in node.keywords if k.arg}
    if len(kwargs) != len(node.keywords):
        raise SandboxError('**kwargs not allowed')
    for v in list(args) + list(kwargs.values()):
        if not isinstance(v, (str, int, float, bool, type(None), list, tuple)):
            raise SandboxError('bad argument')
        if isinstance(v, (list, tuple)) and not all(isinstance(e, (str, int, float, bool)) for e in v):
            raise SandboxError('nested containers not allowed')
    return args, kwargs


def _eval(node, env, allowed):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env, allowed)
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        if isinstance(node.value, int) and not isinstance(node.value, bool) and abs(node.value) > MAX_INT:
            raise SandboxError('integer too large')
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [_eval(e, env, allowed) for e in node.elts]
        return vals if isinstance(node, ast.List) else tuple(vals)
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval(node.operand, env, allowed)
        if not isinstance(v, int):
            raise SandboxError('bad operand')
        return -v
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
        target = _eval(node.value, env, allowed)
        if not isinstance(target, (str, list, tuple)):
            raise SandboxError('bad subscript')
        try:
            return target[node.slice.value]
        except IndexError as e:
            raise ValueError(str(e))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_BUILTINS:
        if node.func.id in env:
            raise SandboxError('shadowed name')
        args, kwargs = _args(node, env, allowed)
        if node.func.id in ('sorted', 'min', 'max') and 'key' in kwargs:
            raise SandboxError('key functions not allowed')
        return _check_size(SAFE_BUILTINS[node.func.id](*args, **kwargs))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id in SAFE_MODULE_FUNCTIONS and node.func.value.id not in env):
        mod, name = node.func.value.id, node.func.attr
        if name not in SAFE_MODULE_FUNCTIONS[mod]:
            raise SandboxError(f'call not allowed: {mod}.{name}')
        args, kwargs = _args(node, env, allowed)
        if name in ('factorial', 'comb', 'perm') and any(isinstance(a, int) and a > 500 for a in args):
            raise SandboxError('argument too large')
        import math, statistics
        return _check_size(getattr({'math': math, 'statistics': statistics}[mod], name)(*args, **kwargs))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        obj = _eval(node.func.value, env, allowed); name = node.func.attr
        if not isinstance(obj, str) or name not in allowed or name not in PURE_STR_METHODS:
            raise SandboxError(f'call not allowed: {type(obj).__name__}.{name}')
        args = [_eval(a, env, allowed) for a in node.args]
        kwargs = {k.arg: _eval(k.value, env, allowed) for k in node.keywords if k.arg}
        if len(kwargs) != len(node.keywords):
            raise SandboxError('**kwargs not allowed')
        for v in list(args) + list(kwargs.values()):
            if not isinstance(v, (str, int, bool, type(None), list, tuple)):
                raise SandboxError('bad argument')
            if name == 'join' and isinstance(v, (list, tuple)) and not all(isinstance(s, str) for s in v):
                raise SandboxError('join needs strings')
        return _check_size(getattr(obj, name)(*args, **kwargs))
    raise SandboxError(f'syntax not allowed: {type(node).__name__}')


def evaluate(source, env=None, allowed=PURE_STR_METHODS):
    """Evaluate one expression string. Raises SandboxError for disallowed code;
    ordinary Python errors (ValueError etc.) are reported as behaviour."""
    if len(source) > 2000:
        raise SandboxError('expression too long')
    try:
        tree = ast.parse(source, mode='eval')
    except SyntaxError as e:
        raise SandboxError(f'not an expression: {e.msg}')
    return _eval(tree, dict(env or {}), allowed)


def run(source, env=None, allowed=PURE_STR_METHODS):
    """Behaviour record: ('ok', value) | ('error', ExceptionName) | ('blocked', reason)."""
    try:
        return ('ok', evaluate(source, env, allowed))
    except SandboxError as e:
        return ('blocked', str(e))
    except (ValueError, TypeError, AttributeError, ArithmeticError) as e:
        return ('error', type(e).__name__)
