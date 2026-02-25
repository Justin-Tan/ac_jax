"""Class for evaluating programs proposed by the Sampler."""
import ast, re, os
import textwrap
import multiprocessing
import concurrent.futures
import asyncio
import traceback

if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from copy import copy
from typing import Sequence, Any, Dict
_SHUTDOWN = "___SHUTDOWN___"

# custom imports
from evolution import code_types, code_utils

class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""
    
    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
        """Collects the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None  # Check internal correctness.
        return self._function_end_line

def _trim_function_body(generated_code: str) -> str:
    """Extracts body of LLM-generated function, trimming postscript."""
    if not generated_code: return ''

    # Strip markdown code blocks if present
    pattern = r"```(?:python|py)?\n(.*?)```"
    match = re.search(pattern, generated_code, re.DOTALL)
    if match:
        generated_code = match.group(1).strip()

    generated_code = textwrap.indent(generated_code, '    ')
    code = f'def dummy_header():\n{generated_code}'
    tree = None
    # We keep trying and deleting code from the end until the parser succeeds.
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            code = '\n'.join(code.splitlines()[:e.lineno - 1])
    if not code:
        # Nothing could be saved from `generated_code`
        return ''
    
    # get body only
    visitor = _FunctionLineVisitor('dummy_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    body_lines = textwrap.dedent('\n'.join(body_lines)).splitlines()
    return '\n'.join(body_lines) + '\n\n'

def _sample_to_program(generated_code: str, version_generated: int | None, template: code_types.Program, 
                      function_to_evolve: str) -> tuple[code_utils.Function, str]:
    """Returns parsed function and python executable as string"""

    # preflight validation
    if not _validate_sample(generated_code):
        raise ValueError(f'Warning: code generated contains questionable commands!\n{generated_code}')

    body = _trim_function_body(generated_code)

    if version_generated is not None:  # rename recursive calls
        body = code_utils.rename_function_calls(body,
            f'{function_to_evolve}_v{version_generated}',
            function_to_evolve)

    # overwrite template body with body from LLM output
    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, program  # str(program)


class SecurityViolation(Exception):
    """Raised when generated code contains security violations."""
    pass

def _validate_sample(sample: str) -> bool:
    # minimal validation, non-comprehensive
    wrapped_code = f"def safety_dummy_header():\n{sample}"
    try:
        tree = ast.parse(wrapped_code)
    except SyntaxError:
        return False
    
    banned_calls = {'eval', 'exec', 'compile', 'open', 'input', 
        '__import__', 'globals', 'locals', 'super', 'getattr', 'setattr'}
    banned_modules = {'os', 'sys', 'subprocess', 'shutil', 'socket', 'pickle',
                      'multiprocessing', 'threading', 'asyncio', 'signal'}

    class SafeASTVisitor(ast.NodeVisitor):
        def visit_Import(self, node) -> None:
            for alias in node.names:
                base_module = alias.name.split('.')[0]
                if base_module in banned_modules:
                    raise SecurityViolation
                
        def visit_ImportFrom(self, node) -> None:
            if node.module and node.module.split('.')[0] in banned_modules:
                raise SecurityViolation
        
        def visit_Call(self, node) -> None:
            if isinstance(node.func, ast.Name):
                if node.func.id in banned_calls:
                    raise SecurityViolation
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {'system', 'popen', 'open', 'spawn'}:
                    raise SecurityViolation
            self.generic_visit(node)

        def visit_Attribute(self, node) -> None:
            if node.attr in {'__globals__', '__code__', '__subclasses__'}:
                raise SecurityViolation
            self.generic_visit(node)

    visitor = SafeASTVisitor()
    try:
        visitor.visit(tree)
        return True
    except SecurityViolation:
        return False