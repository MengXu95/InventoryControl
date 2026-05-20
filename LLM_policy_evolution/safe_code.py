import ast
import types


REQUIRED_FUNCTIONS = (
    'replenishment_policy',
    'transshipment_proposal',
    'consensus_gate',
)

SAFE_BUILTINS = {
    'abs': abs,
    'max': max,
    'min': min,
    'round': round,
}

ALLOWED_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.If,
    ast.Expr,
    ast.Load,
    ast.Store,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


class PolicyValidationError(ValueError):
    pass


class SafePolicyValidator(ast.NodeVisitor):
    def visit(self, node):
        if not isinstance(node, ALLOWED_NODES):
            raise PolicyValidationError(f"Disallowed syntax: {type(node).__name__}")
        return super().visit(node)

    def visit_Module(self, node):
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                raise PolicyValidationError("Only top-level function definitions are allowed.")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name not in REQUIRED_FUNCTIONS:
            raise PolicyValidationError(f"Unexpected function: {node.name}")
        if node.decorator_list:
            raise PolicyValidationError("Decorators are not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_BUILTINS:
            raise PolicyValidationError("Only abs, min, max, and round calls are allowed.")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith('__'):
            raise PolicyValidationError("Dunder names are not allowed.")
        self.generic_visit(node)


def validate_policy_source(source):
    tree = ast.parse(source)
    SafePolicyValidator().visit(tree)
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in REQUIRED_FUNCTIONS if name not in function_names]
    if missing:
        raise PolicyValidationError(f"Missing required functions: {missing}")
    return tree


def load_policy_source(source, filename='<policy>'):
    tree = validate_policy_source(source)
    compiled = compile(tree, filename, 'exec')
    namespace = {'__builtins__': SAFE_BUILTINS}
    exec(compiled, namespace, namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in REQUIRED_FUNCTIONS})


def load_policy_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as policy_file:
        source = policy_file.read()
    policy = load_policy_source(source, filename=str(file_path))
    return policy, source
