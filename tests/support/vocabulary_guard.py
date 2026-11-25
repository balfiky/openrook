"""Source analysis for the perception guards: where does fixed vocabulary live?

Test-only. Detects, in a piece of source, the ways a keyword table can be
written, however it is bound:

* a regular expression that spells out words, whether passed to ``re`` directly,
  bound to a name first, built by concatenation or ``"|".join(...)``, or merely
  present as a pattern-shaped string;
* a collection of natural-language strings (tuple, list, set, dict, nested, or a
  string literal that is ``.split()`` into words) that are not names of the
  runtime's own data model;
* a test of some text against fixed wording (``"word" in x``, ``x == "word"``,
  ``x.startswith("word")`` and similar).

It also resolves which functions a given function calls, so a guard can cover
the helpers of the perception stage and not only its own body.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
import re
from typing import Any, Iterable, NamedTuple

ROOT = Path(__file__).resolve().parents[2]

_JSON_SCHEMA_WORDS = {"type", "items", "properties", "required", "additionalProperties", "enum",
                      "object", "array", "string", "number", "boolean"}
# The two speaker names the conversation history itself uses.
_HISTORY_ROLES = {"user", "assistant"}
# Unicode property values: general categories, their one-letter classes and
# East Asian widths. Code that asks what kind of character something is.
_UNICODE_PROPERTIES = {
    "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf",
    "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
    "L", "M", "N", "P", "S", "Z", "C", "LN", "W", "F", "Na", "H", "A",
}
_REGEX_FUNCTIONS = {"compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split"}
_PROBE_METHODS = {"startswith", "endswith", "find", "rfind", "index", "rindex", "count",
                  "removeprefix", "removesuffix", "partition", "rpartition"}
_COLLECTIONS = (ast.Tuple, ast.List, ast.Set, ast.Dict)


class Finding(NamedTuple):
    kind: str                     # regex | words | probe
    line: int
    detail: str
    terms: tuple[str, ...] = ()   # the literal strings involved


def _has_letters(text: str) -> bool:
    return bool(re.search(r"[^\W\d_]", text))


def data_model_names() -> set[str]:
    """Names the runtime's data model and the reading's schema define. Nothing from prose."""
    from core.appraisal import reading_schema
    from core.types import AppraisalFrame, DetectedEmotion, EventType
    from core.willingness import Evidence, Perception, RequestComponent

    names = set(_JSON_SCHEMA_WORDS) | _HISTORY_ROLES | _UNICODE_PROPERTIES
    # Only the records a reading is made of, not every record in the runtime.
    for record in (AppraisalFrame, DetectedEmotion, Evidence, Perception, RequestComponent):
        names |= {field.name for field in dataclasses.fields(record)}
    names |= {str(member.value) for member in EventType}

    def walk(schema: Any) -> None:
        if isinstance(schema, dict):
            names.update(str(key) for key in schema.get("properties", {}))
            names.update(str(value) for value in schema.get("enum", []))
            for value in schema.values():
                walk(value)
        elif isinstance(schema, list):
            for value in schema:
                walk(value)

    walk(reading_schema(proposals=True))
    return names


def has_vocabulary(pattern: str) -> bool:
    """Whether a regular expression spells out words rather than only structure."""
    bare = re.sub(r"\\[pP]\{[^}]*\}", " ", pattern)                 # \p{..} property classes
    bare = re.sub(r"\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|N\{[^}]*\}|.)", " ", bare)
    bare = re.sub(r"\[[^\]]*\]", " ", bare)                          # character classes
    bare = re.sub(r"\(\?(?:P<\w+>|P=\w+|[a-zA-Z]+\)|[a-zA-Z]+:|[:=!]|<[=!])", " ", bare)  # groups, flags
    return bool(re.search(r"[^\W\d_]{2,}", bare))


def looks_like_pattern(text: str) -> bool:
    """A string written as a regular expression, wherever it ends up being used."""
    return bool(re.search(
        r"\\[bBsSwWdD]"                       # \b, \s, \w, \d and their negations
        r"|\(\?(?:[:=!]|<[=!]|P<|[aiLmsux]+[:)])"  # (?:..), lookarounds, named groups, inline flags
        r"|\[[^\]\s]+\][*+?{]"                 # a character class with a quantifier
        r"|\((?:[^()|\s]+\|)+[^()|\s]+\)",      # a parenthesised alternation of single tokens
        text))


class Scope:
    """Name bindings visible to a piece of code, for following a value to its literals."""

    def __init__(self, *trees: ast.AST) -> None:
        self.bindings: dict[str, list[ast.AST]] = {}
        self.regex_modules: set[str] = set()
        self.regex_functions: set[str] = set()
        for tree in trees:
            self.add(tree)

    def add(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.bindings.setdefault(target.id, []).append(node.value)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name) and node.value:
                self.bindings.setdefault(node.target.id, []).append(node.value)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                self.bindings.setdefault(node.target.id, []).append(node.value)
            elif isinstance(node, ast.Import):
                self.regex_modules |= {alias.asname or alias.name for alias in node.names
                                       if alias.name in {"re", "regex"}}
            elif isinstance(node, ast.ImportFrom) and node.module in {"re", "regex"}:
                self.regex_functions |= {alias.asname or alias.name for alias in node.names
                                         if alias.name in _REGEX_FUNCTIONS}

    def strings(self, node: ast.AST | None, seen: frozenset[str] = frozenset()) -> list[str]:
        """Every string literal an expression may be built from."""
        if node is None:
            return []
        if isinstance(node, ast.Constant):
            return [node.value] if isinstance(node.value, str) else []
        if isinstance(node, ast.Name):
            if node.id in seen:
                return []
            return [text for value in self.bindings.get(node.id, [])
                    for text in self.strings(value, seen | {node.id})]
        return [text for child in ast.iter_child_nodes(node) for text in self.strings(child, seen)]

    def is_regex_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.value.id in self.regex_modules and func.attr in _REGEX_FUNCTIONS
        return isinstance(func, ast.Name) and func.id in self.regex_functions


def _members(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Dict):
        return [item for item in (*node.keys, *node.values) if item is not None]
    return [item.value if isinstance(item, ast.Starred) else item for item in node.elts]


def _held_strings(node: ast.AST) -> list[str]:
    """Strings a literal collection holds itself or through nested literals, not through calls."""
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    if isinstance(node, _COLLECTIONS):
        return [text for member in _members(node) for text in _held_strings(member)]
    return []


def findings(tree: ast.AST, scope: Scope | None = None, *, known: set[str] | None = None,
             strict: bool = True) -> list[Finding]:
    """Every place ``tree`` holds or applies fixed vocabulary.

    ``strict`` also counts an equality test against a literal that is not a
    data-model name. That is right for the perception modules, which have no
    business comparing anything with free wording, and too broad for general
    code, where such literals are mostly the runtime's own labels.
    """
    scope = scope or Scope(tree)
    known = data_model_names() if known is None else known
    found: list[Finding] = []
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)}
    nested = {id(member) for node in ast.walk(tree) if isinstance(node, _COLLECTIONS)
              for member in _members(node) if isinstance(member, _COLLECTIONS)}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        # 1. Regular expressions with words, however the pattern reaches ``re``.
        if scope.is_regex_call(node) and node.args:
            found += [Finding("regex", line, text, (text,)) for text in scope.strings(node.args[0])
                      if has_vocabulary(text)]
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
              and looks_like_pattern(node.value) and has_vocabulary(node.value)):
            found.append(Finding("regex", line, node.value, (node.value,)))
        # 2. Collections of words that are not names from the data model.
        if isinstance(node, _COLLECTIONS) and id(node) not in nested:
            words = [text for text in _held_strings(node) if _has_letters(text)]
            foreign = [text for text in words if text not in known]
            if len(words) >= 2 and foreign:
                found.append(Finding("words", line, ", ".join(foreign[:6]), tuple(foreign)))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "split"
                and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)):
            words = [text for text in node.func.value.value.split() if _has_letters(text)]
            if len(words) >= 2 and any(text not in known for text in words):
                found.append(Finding("words", line, node.func.value.value[:60],
                                     tuple(text for text in words if text not in known)))
        # 3. Tests of text against fixed wording.
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            for position, op in enumerate(node.ops):
                left, right = sides[position], sides[position + 1]
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(left, ast.Constant) \
                        and isinstance(left.value, str) and _has_letters(left.value) and left.value not in known:
                    found.append(Finding("probe", line, left.value, (left.value,)))
                if strict and isinstance(op, (ast.Eq, ast.NotEq)):
                    for side in (left, right):
                        if isinstance(side, ast.Constant) and isinstance(side.value, str) \
                                and _has_letters(side.value) and side.value not in known:
                            found.append(Finding("probe", line, side.value, (side.value,)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _PROBE_METHODS:
            found += [Finding("probe", line, text, (text,)) for argument in node.args
                      for text in _held_strings(argument) if _has_letters(text) and text not in known]
    return list(dict.fromkeys(found))


# ---------------------------------------------------------------------------
# Source units and the call closure of a function
# ---------------------------------------------------------------------------


class Unit(NamedTuple):
    label: str          # "path" or "path::qualname"
    tree: ast.AST       # what is inspected
    scope: Scope        # name bindings it can see
    source: str


_MODULES: dict[str, tuple[str, ast.Module]] = {}


def module_source(path: str | Path) -> tuple[str, ast.Module]:
    key = str(path)
    if key not in _MODULES:
        source = Path(path).read_text(encoding="utf-8")
        _MODULES[key] = (source, ast.parse(source))
    return _MODULES[key]


def module_unit(relative: str, root: Path = ROOT) -> Unit:
    source, tree = module_source(root / relative)
    return Unit(relative, tree, Scope(tree), source)


def _function_node(tree: ast.Module, qualname: str) -> ast.AST | None:
    body: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in qualname.split("."):
        if part == "<locals>":
            return None
        node = next((item for item in body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and item.name == part), None)
        if node is None:
            return None
        body = node.body
    return node


def _module_level(tree: ast.Module) -> ast.Module:
    """The module without its function and class bodies: its constants and imports."""
    return ast.Module(body=[item for item in tree.body
                            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
                      type_ignores=[])


def _referenced(node: ast.AST, tree: ast.Module) -> list[ast.AST]:
    """Module-level assignments a function refers to by name, followed transitively."""
    assignments: dict[str, list[ast.stmt]] = {}
    for item in tree.body:
        targets = item.targets if isinstance(item, ast.Assign) else [item.target] if isinstance(
            item, (ast.AnnAssign, ast.AugAssign)) else []
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(item)
    pending, seen, result = [node], set(), []
    while pending:
        current = pending.pop()
        for name in {n.id for n in ast.walk(current) if isinstance(n, ast.Name)}:
            if name in seen or name not in assignments:
                continue
            seen.add(name)
            result += assignments[name]
            pending += assignments[name]
    return result


def function_unit(function: Any, root: Path = ROOT) -> Unit | None:
    """The source a function runs: its body and the module constants it refers to."""
    function = inspect.unwrap(getattr(function, "__func__", function))
    try:
        path = Path(inspect.getsourcefile(function) or "").resolve()
        relative = path.relative_to(root)
    except (TypeError, ValueError):
        return None
    source, tree = module_source(path)
    node = _function_node(tree, function.__qualname__)
    if node is None:
        return None
    inspected = ast.Module(body=[node, *_referenced(node, tree)], type_ignores=[])
    return Unit(f"{relative}::{function.__qualname__}", inspected, Scope(_module_level(tree), node), source)


def _local_imports(node: ast.AST) -> dict[str, Any]:
    names: dict[str, Any] = {}
    for item in ast.walk(node):
        try:
            if isinstance(item, ast.ImportFrom) and item.module and not item.level:
                module = importlib.import_module(item.module)
                names.update({alias.asname or alias.name: getattr(module, alias.name) for alias in item.names})
            elif isinstance(item, ast.Import):
                names.update({(alias.asname or alias.name).split(".")[0]:
                              importlib.import_module(alias.name.split(".")[0]) for alias in item.names})
        except (ImportError, AttributeError):
            continue
    return names


def _resolve(expression: ast.AST, namespace: dict[str, Any]) -> Any:
    chain = []
    while isinstance(expression, ast.Attribute):
        chain.append(expression.attr)
        expression = expression.value
    if not isinstance(expression, ast.Name) or expression.id not in namespace:
        return None
    value = namespace[expression.id]
    for attribute in reversed(chain):
        try:
            value = getattr(value, attribute)
        except Exception:
            return None
    return value


def callees(function: Any, *, instance: Any = None, root: Path = ROOT) -> list[Any]:
    """Functions under ``root`` that ``function`` calls, resolved against live objects.

    ``instance`` stands for ``self`` when ``function`` is a method of its class,
    so attribute chains such as ``self.memory.retrieve`` resolve to real code.
    """
    plain = inspect.unwrap(getattr(function, "__func__", function))
    unit = function_unit(plain, root)
    if unit is None:
        return []
    node = unit.tree.body[0]
    module = inspect.getmodule(plain)
    namespace = dict(vars(module)) if module else {}
    namespace.update(_local_imports(node))
    owner_name = plain.__qualname__.split(".")[0]
    owner = namespace.get(owner_name) if "." in plain.__qualname__ else None
    if isinstance(owner, type):
        namespace["self"] = instance if isinstance(instance, owner) else owner
    found = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        target = _resolve(item.func, namespace)
        target = inspect.unwrap(getattr(target, "__func__", target)) if target is not None else None
        if inspect.isfunction(target) and function_unit(target, root) is not None and target is not plain:
            found.append(target)
    return list(dict.fromkeys(found))


def call_closure(functions: Iterable[Any], *, instance: Any = None, depth: int = 3, root: Path = ROOT) -> list[Any]:
    """The given functions and everything they call, ``depth`` calls deep."""
    order: dict[Any, int] = {}
    frontier = [(inspect.unwrap(getattr(f, "__func__", f)), 0) for f in functions]
    while frontier:
        function, level = frontier.pop(0)
        if function in order:
            continue
        order[function] = level
        if level < depth:
            frontier += [(callee, level + 1) for callee in callees(function, instance=instance, root=root)]
    return list(order)


def label(function: Any, root: Path = ROOT) -> str:
    unit = function_unit(function, root)
    return unit.label if unit else repr(function)


# ---------------------------------------------------------------------------
# Which functions does a person's text reach?
# ---------------------------------------------------------------------------


def _carries(value: Any, marker: str) -> bool:
    """The text itself, a short sequence of texts, or a short conversation history."""
    if isinstance(value, str):
        return marker in value.casefold()
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) <= 64:
        return any(isinstance(item, str) and marker in item.casefold()
                   or isinstance(item, dict) and any(isinstance(part, str) and marker in part.casefold()
                                                     for part in item.values())
                   for item in value)
    return False


class TextTrace:
    """Records every function under ``root`` that is called with the marked text.

    Used as a context manager around real turns whose message contains
    ``marker``. A function counts when the text arrives in one of its
    arguments, directly or inside a short list or mapping, in any letter case.
    """

    def __init__(self, marker: str, root: Path = ROOT, exclude: tuple[str, ...] = ("tests",)) -> None:
        self.marker = marker.casefold()
        self.root = root
        self.exclude = exclude
        self.reached: dict[tuple[str, str], None] = {}
        self._previous: Any = None

    def _profile(self, frame: Any, event: str, argument: Any) -> None:
        if event != "call":
            return
        code = frame.f_code
        if "<" in code.co_qualname:
            return  # comprehensions, lambdas and nested helpers belong to their function
        try:
            relative = Path(code.co_filename).resolve().relative_to(self.root)
        except ValueError:
            return
        if relative.parts[0] in self.exclude:
            return
        names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
        extra = code.co_varnames[len(names):len(names) + bool(code.co_flags & 0x04) + bool(code.co_flags & 0x08)]
        if any(_carries(frame.f_locals.get(name), self.marker) for name in (*names, *extra)):
            self.reached[(str(relative), code.co_qualname)] = None

    def __enter__(self) -> "TextTrace":
        import sys

        self._previous = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *_: Any) -> None:
        import sys

        sys.setprofile(self._previous)

    def units(self) -> list[Unit]:
        """The reached functions as inspectable source units."""
        result = []
        for relative, qualname in self.reached:
            source, tree = module_source(self.root / relative)
            node = _function_node(tree, qualname)
            if node is None or isinstance(node, ast.ClassDef):
                continue
            inspected = ast.Module(body=[node, *_referenced(node, tree)], type_ignores=[])
            result.append(Unit(f"{relative}::{qualname}", inspected, Scope(_module_level(tree), node), source))
        return result
