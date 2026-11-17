"""Keep native AlignIns selection compatible with NumPy scalar conversion rules.

Only replace its two np.argwhere calls by np.flatnonzero. For these 1-D masks,
this returns the same selected indices as the old int(array([index])) code.
The original source is unchanged so queued experiments retain their source hashes.
"""
import ast
import inspect
import textwrap


def apply_numpy_compat():
    from aggregation import Aggregation
    original = Aggregation.agg_alignins
    if getattr(original, '_numpy_index_compat', False):
        return
    source, first_line = inspect.getsourcelines(original)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'np' and node.func.attr == 'argwhere'):
            node.func.attr = 'flatnonzero'
            count += 1
    if count != 2:
        raise RuntimeError(f'Expected two native AlignIns index extractions, found {count}')
    ast.increment_lineno(tree, first_line - 1)
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), 'exec'),
         original.__globals__, namespace)
    replacement = namespace['agg_alignins']
    replacement._numpy_index_compat = True
    Aggregation.agg_alignins = replacement
