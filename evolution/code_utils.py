import jax
import jax.numpy as jnp
import ast

import dataclasses
import io, re
import tokenize
from typing import Sequence, Any
from collections.abc import Iterator, MutableSet, Sequence

from ac_jax import logging
from evolution import code_types


def text_to_program(text: str) -> code_types.Program:
    """Returns Program object by parsing input text using Python AST."""
    # Strip markdown code blocks if present
    pattern = r"```(?:python|py)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        # We assume that the program is composed of some preface (e.g. imports,
        # classes, assignments, ...) followed by a sequence of functions.
        tree = ast.parse(text)
        visitor = code_types.ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()
    except Exception as e:
        logging.warning('Failed parsing %s', text)
        raise e

def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    """Transforms `code` into Python tokens."""
    code_bytes = code.encode()
    code_io = io.BytesIO(code_bytes)
    return tokenize.tokenize(code_io.readline)

def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    """Transforms a list of Python tokens into code."""
    code_bytes = tokenize.untokenize(tokens)
    return code_bytes.decode()


def _yield_token_and_is_call(
    code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
  """Yields each token with a bool indicating whether it is a function call."""
  try:
    tokens = _tokenize(code)
    prev_token = None
    is_attribute_access = False
    for token in tokens:
      if (prev_token and  # If the previous token exists and
          prev_token.type == tokenize.NAME and  # it is a Python identifier
          token.type == tokenize.OP and  # and the current token is a delimiter
          token.string == '('):  # and in particular it is '('.
        yield prev_token, not is_attribute_access
        is_attribute_access = False
      else:
        if prev_token:
          is_attribute_access = (
              prev_token.type == tokenize.OP and prev_token.string == '.'
          )
          yield prev_token, False
      prev_token = token
    if prev_token:
      yield prev_token, False
  except Exception as e:
    logging.warning('Failed parsing %s', code)
    raise e


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """Renames function calls from `source_name` to `target_name`."""
    if source_name not in code: return code
    modified_tokens = []
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            # Replace the function name token
            modified_token = tokenize.TokenInfo(
                type=token.type,
                string=target_name,
                start=token.start,
                end=token.end,
                line=token.line,
            )
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
  """Returns the set of all functions called in `code`."""
  return set(token.string for token, is_call in
             _yield_token_and_is_call(code) if is_call)

def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Checks if generated function is calling an earlier version."""
    for name in get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False

