import ast

from chex import dataclass, PRNGKey
from typing import Any, Dict, NamedTuple, Optional, Union, List

class Heuristics(NamedTuple):
    code: str
    score: float

class ProgramDatabase(NamedTuple):
    programs: List[Heuristics]
    islands: int

class Prompt(NamedTuple):
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: Prompt, ending with header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  """
  code: str
  version_generated: int
  island_id: int


@dataclass
class Function:
    """A parsed Python function."""
    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None

    def __str__(self) -> str:
        return_type = f' -> {self.return_type}' if self.return_type else ''

        function = f'def {self.name}({self.args}){return_type}:\n'
        if self.docstring:
            # self.docstring is already indented on every line except the first one.
            # Here, we assume the indentation is always two spaces.
            new_line = '\n' if self.body else ''
            function += f'  """{self.docstring}"""{new_line}'
        # self.body is already indented.
        function += self.body + '\n\n'
        return function

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`.
        if name == 'body':
            value = value.strip('\n')
        # Ensure there aren't leading & trailing quotes in `docstring``.
        if name == 'docstring' and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', '')
        object.__setattr__(self, name, value)

@dataclass
class Program:
    """A parsed Python program possibly consisting of multiple functions."""
    # `preface` is everything from the beginning of the code till the first
    # function is found.
    preface: str
    functions: list[Function]

    def __str__(self) -> str:
        program = f'{self.preface}\n' if self.preface else ''
        program += '\n'.join([str(f) for f in self.functions])
        return program

    def find_function_index(self, function_name: str) -> int:
        """Returns the index of input function name."""
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            raise ValueError(
                f'function {function_name} does not exist in program:\n{str(self)}'
            )
        if count > 1:
            raise ValueError(
                f'function {function_name} exists more than once in program:\n'
                f'{str(self)}'
            )
        index = function_names.index(function_name)
        return index

    def get_function(self, function_name: str) -> Function:
        index = self.find_function_index(function_name)
        return self.functions[index]

class Task(NamedTuple):
    """Input to evaluation pipeline."""
    candidate_id: int
    program: Program
    function_to_run: str
    seed: int  # seed for random key

class ProgramVisitor(ast.NodeVisitor):
    """Parses code to collect all required information to produce a `Program`.

    Note that we do not store function decorators.
    """

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()

        self._preface: str = ''
        self._functions: list[Function] = []
        self._current_function: str | None = None

    def visit_FunctionDef(self,  # pylint: disable=invalid-name
                            node: ast.FunctionDef) -> None:
        """Collects all information about the function being parsed."""
        if node.col_offset == 0:  # We only care about first level functions.
            self._current_function = node.name
            if not self._functions:
                self._preface = '\n'.join(self._codelines[:node.lineno - 1])
            function_end_line = node.end_lineno
            body_start_line = node.body[0].lineno - 1
            # Extract the docstring.
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value,
                                                                ast.Str):
                docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
                if len(node.body) > 1:
                    body_start_line = node.body[1].lineno - 1
                else:
                    body_start_line = function_end_line

            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args),
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body='\n'.join(self._codelines[body_start_line:function_end_line]),
            ))
        self.generic_visit(node)


    def return_program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)


# Sentinel for heuristics pub/sub queue shutdown
HEURISTICS_QUEUE_SHUTDOWN = object()
