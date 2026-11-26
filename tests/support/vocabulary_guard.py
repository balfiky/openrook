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

When it is told which expressions hold the user's text (:class:`TextFlow`), a
test of that text counts whatever the literal is, including names from the data
model, and so does a lookup of the text in a table of any size.

:class:`TextTrace` supplies that knowledge from real turns: it records every
function the marked text is handed to, as a string or inside the runtime's own
objects, under which names and fields, and with which module globals and
owning object, so a table or pattern that is imported from another module, kept
on the object, or built when a module loads is seen where it is applied.
It also resolves which functions a given function calls.

Known limit: a table built inside the function at call time, or handed in as
an argument, has no value to follow at its point of use; its words are still
found wherever they are written as literals.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
import re
import types
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
    on_text: bool = False         # the literal is tested against the user's text


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


_LITERAL_MEMBERS = 4000


def _as_literal(value: Any, budget: list[int] | None = None) -> ast.AST | None:
    """A runtime value written back as the literal it would be in source, when it is plain data.

    Strings, compiled patterns (as their pattern text) and collections of plain
    data qualify. Anything else has no literal form and yields None.
    """
    budget = [_LITERAL_MEMBERS] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        return None
    if isinstance(value, re.Pattern):
        return ast.Constant(value.pattern) if isinstance(value.pattern, str) else None
    if isinstance(value, str):
        return ast.Constant(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        members = [_as_literal(item, budget) for item in value]
        return ast.Tuple(elts=[item for item in members if item is not None], ctx=ast.Load())
    if isinstance(value, dict):
        pairs = [(_as_literal(key, budget), _as_literal(item, budget)) for key, item in value.items()]
        members = [item for pair in pairs for item in pair if item is not None]
        return ast.Tuple(elts=members, ctx=ast.Load())
    return None


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
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        self.bindings.setdefault(target.id, []).append(node.iter)
            elif isinstance(node, ast.Import):
                self.regex_modules |= {alias.asname or alias.name for alias in node.names
                                       if alias.name in {"re", "regex"}}
            elif isinstance(node, ast.ImportFrom) and node.module in {"re", "regex"}:
                self.regex_functions |= {alias.asname or alias.name for alias in node.names
                                         if alias.name in _REGEX_FUNCTIONS}

    def bind_runtime(self, node: ast.AST, namespace: dict[str, Any], owner: Any = None) -> None:
        """Bind what the code uses to the values it really has, beside what the source shows.

        A word list or pattern imported from another module, reached as
        ``module.NAME`` or ``self.NAME``, or built by a call when its module was
        loaded, has no literal in the inspected source. Its runtime value is
        written back as one, so a test of the text against it is seen.
        ``namespace`` is the function's module globals and ``owner`` the object
        it was called on, if any. Attributes are read without running any code.
        """
        local = {argument.arg for item in ast.walk(node) if isinstance(item, ast.arguments)
                 for argument in (*item.posonlyargs, *item.args, *item.kwonlyargs, item.vararg, item.kwarg)
                 if argument is not None} | set(Scope(node).bindings)
        bound: set[str] = set()
        for item in ast.walk(node):
            key, value = None, None
            if isinstance(item, ast.Name) and item.id not in local and item.id in namespace:
                key, value = item.id, namespace[item.id]
            elif isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
                holder = owner if item.value.id in ("self", "cls") else (
                    namespace.get(item.value.id) if item.value.id not in local else None)
                if isinstance(holder, types.ModuleType) or (holder is not None and holder is owner):
                    key, value = f"{item.value.id}.{item.attr}", inspect.getattr_static(holder, item.attr, None)
            if key is None or key in bound:
                continue
            bound.add(key)
            literal = _as_literal(value)
            if literal is not None:
                self.bindings.setdefault(key, []).append(literal)

    @staticmethod
    def _dotted(node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return None

    def strings(self, node: ast.AST | None) -> list[str]:
        """Every string literal an expression may be built from.

        Each name is followed once, so the cost stays linear in the size of the
        code however densely its names refer to one another.
        """
        found: list[str] = []
        visited: set[str] = set()
        pending = [node]
        while pending:
            item = pending.pop()
            if item is None:
                continue
            if isinstance(item, ast.Constant):
                if isinstance(item.value, str):
                    found.append(item.value)
            elif isinstance(item, ast.Name):
                if item.id not in visited:
                    visited.add(item.id)
                    pending.extend(reversed(self.bindings.get(item.id, [])))
            elif self._dotted(item) in self.bindings:
                dotted = self._dotted(item)
                if dotted not in visited:
                    visited.add(dotted)
                    pending.extend(reversed(self.bindings[dotted]))
            else:
                pending.extend(reversed(list(ast.iter_child_nodes(item))))
        return found

    def table(self, node: ast.AST | None) -> list[str]:
        """The strings a collection is made of, following names to where they were bound.

        Unlike :meth:`strings` this leaves out what is not content of the
        collection: the key of a field read (``record.get("context")``,
        ``record["context"]``), the literal of a comparison, and formatted text.
        """
        found: list[str] = []
        visited: set[str] = set()
        pending = [node]
        while pending:
            item = pending.pop()
            if self._dotted(item) in self.bindings:
                dotted = self._dotted(item)
                if dotted not in visited:
                    visited.add(dotted)
                    pending.extend(reversed(self.bindings[dotted]))
                continue
            if item is None or isinstance(item, (ast.Compare, ast.JoinedStr, ast.Lambda, ast.Attribute)):
                continue
            if isinstance(item, ast.Constant):
                if isinstance(item.value, str):
                    found.append(item.value)
            elif isinstance(item, ast.Name):
                if item.id not in visited:
                    visited.add(item.id)
                    pending.extend(reversed(self.bindings.get(item.id, [])))
            elif isinstance(item, ast.Subscript):
                pending.append(item.value)
            elif isinstance(item, ast.Call):
                if isinstance(item.func, ast.Attribute):
                    if item.func.attr == "get":
                        continue
                    receiver = item.func.value
                    if item.func.attr == "split" and isinstance(receiver, ast.Constant) and isinstance(
                            receiver.value, str):
                        found.extend(receiver.value.split())
                        continue
                    pending.append(receiver)
                pending.extend(reversed([*item.args, *(keyword.value for keyword in item.keywords)]))
            else:
                pending.extend(reversed(list(ast.iter_child_nodes(item))))
        return found

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


_TEXT_METHODS_ON_PATTERN = {"search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split"}
# Calls whose result says something about text without being text.
_MEASURING_CALLS = {
    "len", "ord", "bool", "int", "float", "any", "all", "sum", "isinstance", "hash", "id", "type", "range",
    "hasattr", "callable", "category", "east_asian_width", "isspace", "isalpha", "isdigit", "isalnum",
    "isupper", "islower", "isnumeric", "startswith", "endswith", "count", "find", "rfind", "index", "rindex",
}


def _bound(target: ast.AST) -> list[str]:
    """The names an assignment target gives a value to.

    ``table[key] = text`` puts text into ``table``; it says nothing about ``key``.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _bound(item)]
    if isinstance(target, (ast.Starred, ast.Subscript)):
        return _bound(target.value)
    return []


class TextFlow:
    """Which expressions inside a piece of code hold the user's text.

    ``names`` are parameters that received the text itself. ``leaves`` are the
    attribute and key names under which the text was seen travelling inside
    objects (``user_message``, ``description``, ``content`` and so on). Anything
    assigned or iterated from such an expression holds text too.
    """

    def __init__(self, tree: ast.AST, names: Iterable[str] = (), leaves: Iterable[str] = ()) -> None:
        self.names = set(names)
        self.leaves = set(leaves)
        changed = bool(self.names or self.leaves)
        while changed:
            changed = False
            for node in ast.walk(tree):
                value, targets = None, []
                if isinstance(node, ast.Assign):
                    value, targets = node.value, node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                    value, targets = node.value, [node.target]
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    value, targets = node.iter, [node.target]
                elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                    value, targets = node.context_expr, [node.optional_vars]
                if value is None or not self.holds(value):
                    continue
                for target in targets:
                    for name in _bound(target):
                        if name not in self.names:
                            self.names.add(name)
                            changed = True

    def holds(self, node: ast.AST | None) -> bool:
        """Whether an expression is, or is computed from, the user's text.

        Field-sensitive: reading a named field or key yields text only when the
        text was seen under that name, so ``st.user_message`` and
        ``component.description`` hold text while ``component.kind`` and
        ``item["role"]`` do not. A comparison yields a truth value and a
        measuring call (``len``, a character's Unicode category) yields a
        number or a code, not text.
        """
        if node is None or not (self.names or self.leaves):
            return False
        if isinstance(node, ast.Name):
            return node.id in self.names
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.Compare):
            return False
        if isinstance(node, ast.Attribute):
            return node.attr in self.leaves
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value in self.leaves
            return self.holds(node.value)
        if isinstance(node, ast.Call):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method == "get" and node.args and isinstance(node.args[0], ast.Constant):
                    return node.args[0].value in self.leaves
                if method in _MEASURING_CALLS:
                    return False
                return self.holds(node.func.value) or any(self.holds(argument) for argument in arguments)
            if isinstance(node.func, ast.Name) and node.func.id in _MEASURING_CALLS:
                return False
            return any(self.holds(argument) for argument in arguments)
        return any(self.holds(child) for child in ast.iter_child_nodes(node))


def _lettered(texts: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text for text in texts if _has_letters(text)))


def findings(tree: ast.AST, scope: Scope | None = None, *, known: set[str] | None = None,
             strict: bool = True, text_names: Iterable[str] = (), text_leaves: Iterable[str] = ()) -> list[Finding]:
    """Every place ``tree`` holds or applies fixed vocabulary.

    Three kinds of finding. The first two do not depend on what the code is
    applied to; the third is about tests of text:

    * ``regex``: a pattern that spells out words, however it reaches ``re``.
    * ``words``: a collection of two or more strings that are not names from
      the data model.
    * ``probe``: a fixed string tested against something. When that something
      is the user's text (see :class:`TextFlow`), every literal counts, whether
      or not it is also a data-model name, and so does a lookup of the text in
      a literal table of any size: the allow-list excuses comparisons with the
      reading's own fields, never with what a person wrote. Elsewhere a literal
      counts when it is not a data-model name; ``strict`` extends that to
      equality tests.
    """
    scope = scope or Scope(tree)
    known = data_model_names() if known is None else known
    flow = TextFlow(tree, text_names, text_leaves)
    found: list[Finding] = []
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)}
    nested = {id(member) for node in ast.walk(tree) if isinstance(node, _COLLECTIONS)
              for member in _members(node) if isinstance(member, _COLLECTIONS)}

    def literal(node: ast.AST) -> str | None:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        # 1. Regular expressions with words, however the pattern reaches ``re``.
        if scope.is_regex_call(node) and node.args:
            applied = any(flow.holds(argument) for argument in
                          [*node.args[1:], *(keyword.value for keyword in node.keywords)])
            found += [Finding("regex", line, text, (text,), applied) for text in scope.strings(node.args[0])
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
        # 3. Tests of something against fixed wording.
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            for position, op in enumerate(node.ops):
                left, right = sides[position], sides[position + 1]
                if isinstance(op, (ast.In, ast.NotIn)):
                    word = literal(left)
                    if word is not None and _has_letters(word) and (flow.holds(right) or word not in known):
                        found.append(Finding("probe", line, word, (word,), flow.holds(right)))
                    table = _lettered(scope.table(left)) if flow.holds(right) and word is None else ()
                    if table:   # ``for word in TABLE: word in text``
                        found.append(Finding("probe", line, ", ".join(table[:6]), table, True))
                    table = _lettered(scope.table(right)) if flow.holds(left) else ()
                    if table:   # ``text in TABLE``, of any size
                        found.append(Finding("probe", line, ", ".join(table[:6]), table, True))
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    for side, other in ((left, right), (right, left)):
                        word = literal(side)
                        if word is None or not _has_letters(word):
                            continue
                        if flow.holds(other) or (strict and word not in known):
                            found.append(Finding("probe", line, word, (word,), flow.holds(other)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver, method = node.func.value, node.func.attr
            on_text = flow.holds(receiver)
            if method in _PROBE_METHODS:
                found += [Finding("probe", line, text, (text,), on_text) for argument in node.args
                          for text in _held_strings(argument)
                          if _has_letters(text) and (on_text or text not in known)]
            if method == "get" and node.args and flow.holds(node.args[0]):
                table = _lettered(scope.table(receiver))
                if table:   # ``TABLE.get(text)``, of any size
                    found.append(Finding("probe", line, ", ".join(table[:6]), table, True))
            if method in _TEXT_METHODS_ON_PATTERN and any(flow.holds(argument) for argument in node.args):
                found += [Finding("regex", line, text, (text,), True) for text in scope.strings(receiver)
                          if has_vocabulary(text)]
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant) and flow.holds(node.slice):
            table = _lettered(scope.table(node.value))
            if table:       # ``TABLE[text]``, of any size
                found.append(Finding("probe", line, ", ".join(table[:6]), table, True))
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


# Fields that hold what a person wrote wherever they appear in a traced
# function: the turn's message and the text of a request component.
CORE_TEXT_FIELDS = ("user_message", "description", "source_span", "role_identity")
_INSPECTED_PACKAGES = ("core", "pipeline", "runtime", "openrook_tools", "interface", "config", "evals")
_NODE_BUDGET = 1500
DIRECT = ""  # the text itself, as opposed to a field that holds it


def _carriers(value: Any, marker: str, depth: int, budget: list[int], leaf: str = DIRECT,
              packages: tuple[str, ...] = _INSPECTED_PACKAGES) -> set[str]:
    """Under which names the marked text sits inside ``value``.

    The text itself reports :data:`DIRECT`. Inside a mapping or one of the
    runtime's own objects it reports the key or attribute that holds it, so
    the turn state yields ``user_message`` and a request component yields
    ``description`` and ``source_span``.
    """
    budget[0] -= 1
    if budget[0] < 0:
        return set()
    if isinstance(value, str):
        return {leaf} if marker in value.casefold() else set()
    if depth <= 0 or value is None or isinstance(value, (int, float, bool, bytes)):
        return set()
    found: set[str] = set()
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in list(value)[:64]:
            found |= _carriers(item, marker, depth - 1, budget, leaf, packages)
    elif isinstance(value, dict):
        for key, item in list(value.items())[:64]:
            found |= _carriers(item, marker, depth - 1, budget, key if isinstance(key, str) else leaf, packages)
    elif type(value).__module__.split(".")[0] in packages:
        if dataclasses.is_dataclass(value):
            members = [(field.name, getattr(value, field.name, None)) for field in dataclasses.fields(value)]
        else:
            members = list(getattr(value, "__dict__", {}).items())
        for name, item in members[:64]:
            found |= _carriers(item, marker, depth - 1, budget, name, packages)
    return found


class TextTrace:
    """Records every function under ``root`` that is handed the marked text.

    Used as a context manager around real turns whose message contains
    ``marker``. A function counts when the text arrives in one of its
    arguments: as a string, in a sequence or mapping, or inside one of the
    runtime's own objects such as the turn state or a request component. For
    each function it keeps which parameters held the text itself and under
    which field names it travelled, so the code can then be checked for tests
    against exactly those expressions. ``self`` is not followed: what an object
    remembers between calls is not an argument.
    """

    def __init__(self, marker: str, root: Path = ROOT, exclude: tuple[str, ...] = ("tests",),
                 depth: int = 5, packages: tuple[str, ...] = _INSPECTED_PACKAGES) -> None:
        self.marker = marker.casefold()
        self.root = root
        self.exclude = exclude
        self.depth = depth
        self.packages = packages
        self.reached: dict[tuple[str, str], dict[str, set[str]]] = {}
        self._namespaces: dict[tuple[str, str], dict[str, Any]] = {}
        self._owners: dict[tuple[str, str], Any] = {}
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
        count = code.co_argcount + code.co_kwonlyargcount + bool(code.co_flags & 0x04) + bool(code.co_flags & 0x08)
        direct, leaves = set(), set()
        for name in code.co_varnames[:count]:
            if name in ("self", "cls"):
                continue
            held = _carriers(frame.f_locals.get(name), self.marker, self.depth, [_NODE_BUDGET],
                             packages=self.packages)
            if DIRECT in held:
                direct.add(name)
            leaves |= held - {DIRECT}
        if direct or leaves:
            key = (str(relative), code.co_qualname)
            record = self.reached.setdefault(key, {"names": set(), "leaves": set()})
            record["names"] |= direct
            record["leaves"] |= leaves
            self._namespaces[key] = frame.f_globals
            if code.co_varnames[:1] in (("self",), ("cls",)):
                self._owners[key] = frame.f_locals.get(code.co_varnames[0])

    @property
    def leaves(self) -> set[str]:
        """Every field name the text was seen travelling under."""
        return {leaf for record in self.reached.values() for leaf in record["leaves"]}

    def __enter__(self) -> "TextTrace":
        import sys

        self._previous = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *_: Any) -> None:
        import sys

        sys.setprofile(self._previous)

    def units(self) -> list[tuple[Unit, set[str], set[str]]]:
        """The reached functions as source units, each with its text parameters and text fields."""
        result = []
        for (relative, qualname), record in self.reached.items():
            source, tree = module_source(self.root / relative)
            node = _function_node(tree, qualname)
            if node is None or isinstance(node, ast.ClassDef):
                continue
            inspected = ast.Module(body=[node, *_referenced(node, tree)], type_ignores=[])
            scope = Scope(_module_level(tree), node)
            scope.bind_runtime(node, self._namespaces.get((relative, qualname), {}),
                               self._owners.get((relative, qualname)))
            unit = Unit(f"{relative}::{qualname}", inspected, scope, source)
            result.append((unit, set(record["names"]), set(record["leaves"]) | set(CORE_TEXT_FIELDS)))
        return result

    def findings(self, *, known: set[str] | None = None) -> dict[str, list[Finding]]:
        """What each reached function holds or applies, checked strictly against the text it was handed.

        A function's text is whatever arrived as text in its parameters, the
        fields its own arguments carried the text under, and the message and
        component text fields wherever it reads them.
        """
        known = data_model_names() if known is None else known
        flagged = {}
        for unit, names, leaves in self.units():
            found = findings(unit.tree, unit.scope, known=known, strict=True, text_names=names, text_leaves=leaves)
            if found:
                flagged[unit.label] = found
        return flagged
