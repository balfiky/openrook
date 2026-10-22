"""Ground explicit filesystem requests in literal operands, before choice.

This is request recognition, not authorization. Unsupported/ambiguous phrasing
stays with semantic appraisal; recognized operations still need atomic grants
and the ordinary operator/tool gates.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath
import re

from core.willingness import RequestComponent

_START = re.compile(
    r"(?:^|(?<=[.!?;])\s+|,\s*|\b(?:and|then)\s+)"
    r"(?:please\s+|as an? optional favor,\s*)?"
    r"(?P<action>create|write|save|read|list|delete)\s+", re.I,
)
_QUOTES = re.compile(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|(?<!\w)\'[^\'\n]*\'')
_PATH = r'(`[^`]+`|"[^"\n]+"|\'[^\'\n]+\'|[^\s,;]+)'
_VERBS = {"fs.write_file": r"\b(?:create|writ\w*|wrote|save)\b",
          "fs.read_file": r"\bread\b", "fs.list_dir": r"\blist\b",
          "fs.delete_path": r"\b(?:delete|remove)\b"}


def _path(raw: str) -> str:
    value = raw.strip('`"\'').rstrip(".!?")
    return value if "/" in value or "." in value else ""


def file_requests(text: str) -> tuple[RequestComponent, ...]:
    masked = _QUOTES.sub(lambda m: " " * len(m.group()), text)
    starts = list(_START.finditer(masked))
    if starts and re.search(r"\b(?:summari[sz]e|translate|quote|analy[sz]e)\b",
                            masked[:starts[0].start()], re.I):
        return ()
    exact_tail = re.search(r"\bexactly this content:\s*", text, re.I)
    if exact_tail:
        starts = [m for m in starts if m.start() < exact_tail.end()]
    result, previous = [], ""
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[match.end():end]
        action = match["action"].lower()
        path, args, tool = "", {}, ""
        operand = re.match(r"(?:(?:the|a|only)\s+)?(?:file\s+)?" + _PATH, body)
        if operand:
            path = _path(operand[1])
        if action in {"create", "write", "save"} and path:
            content = re.search(r"\b(?:with\s+(?:exactly\s+)?(?:this\s+)?content\s*:?|containing)\s*", body, re.I)
            if content:
                literal = body[content.end():]
                # The end of an operation can be followed by a report/condition
                # clause. Neither is part of the requested file bytes.
                literal = re.split(r"\.\s+(?:If you|Report each)\b", literal, maxsplit=1, flags=re.I)[0]
                if i + 1 < len(starts):
                    literal = literal.rstrip(" ,")
                if len(literal) >= 2 and literal[0] in "`\"'" and literal[-1] == literal[0]:
                    literal = literal[1:-1]
                args, tool = {"path": path, "content": literal}, "fs.write_file"
        elif action == "read":
            if re.match(r"it\s+back\b", body, re.I):
                path = previous
            if path:
                args, tool = {"path": path}, "fs.read_file"
        elif action == "list":
            if previous and re.search(r"\b(?:its|the) parent directory\b", body, re.I):
                path = str(PurePosixPath(previous).parent)
            else:
                operand = re.match(r"(?:the\s+)?(?:entries in\s+|files in\s+|directory\s+)?" + _PATH, body, re.I)
                path = _path(operand[1]) if operand else ""
            if path:
                args, tool = {"path": path}, "fs.list_dir"
        elif action == "delete" and path:
            args, tool = {"path": path}, "fs.delete_path"
        if not tool:
            continue
        if tool != "fs.list_dir":
            previous = path
        source = text[match.start("action"):end].strip()
        result.append(RequestComponent(
            f"file{i}", source, user_value=0.6, effort=0.12, uncertainty=0.1,
            source_span=source, requested_tool=tool,
            requested_arguments=tuple(args.items()), capability_requirements=(tool,),
            evidence_refs=(f"literal:file{i}",), confidence=0.9,
        ))
    return tuple(result)


def ground_file_components(text: str, components: tuple[RequestComponent, ...]) -> tuple[RequestComponent, ...]:
    literal = file_requests(text)
    if not literal:
        return components
    remaining, grounded = list(components), []
    def refers_to(c, request):
        path = dict(request.requested_arguments)["path"]
        name = PurePosixPath(path).name
        description = c.description.lower()
        return (path.lower() in description or bool(name and name.lower() in description)
                or bool(re.search(r"\b(?:it back|its parent|parent directory|the file|do not create it)\b", description)))
    for request in literal:
        modeled = next((c for c in remaining if c.kind == "task" and
                        re.search(_VERBS[request.requested_tool], c.description, re.I)
                        and refers_to(c, request)), None)
        if modeled:
            remaining.remove(modeled)
            request = replace(modeled, description=request.description, source_span=request.source_span,
                requested_tool=request.requested_tool, requested_arguments=request.requested_arguments,
                capability_requirements=request.capability_requirements,
                evidence_refs=(*modeled.evidence_refs, *request.evidence_refs))
        grounded.append(request)
    # A combined modeled read/list request is replaced by its atomic operations.
    # Other independently answerable requests and roles retain their consequences.
    remaining = [c for c in remaining if c.kind != "task" or not any(
        re.search(_VERBS[r.requested_tool], c.description, re.I) and refers_to(c, r) for r in literal)]
    remaining = [replace(c, kind="conversation") if re.match(r"report\b", c.description, re.I) else c for c in remaining]
    combined = [c for c in remaining if c.kind == "role"] + grounded + [c for c in remaining if c.kind != "role"]
    return tuple(replace(c, id=f"c{i}") for i, c in enumerate(combined[:8]))


def ground_shell_components(text: str, components: tuple[RequestComponent, ...]) -> tuple[RequestComponent, ...]:
    """Bind an explicitly delimited command before choice, without selecting one.

    Ambiguous prose and compound instructions stay with appraisal. The literal
    command receives the same character grant and operator checks as any task.
    """
    match = re.fullmatch(
        r"\s*(?:please\s+)?run (?:this|the) [^:\n]{0,160}\b(?:shell|command)\b[^:\n]*:\s*(.+?)\s*",
        text, re.I | re.S)
    if not match:
        return components
    command = match[1]
    if command.startswith("```"):
        fenced = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", command, re.S)
        if not fenced:
            return components
        command = fenced[1]
    elif command.startswith("`") and command.endswith("`"):
        command = command[1:-1]
    # A following natural-language sentence is not silently executed as shell.
    if not command.strip() or "\n" in command or re.search(r"\.\s+[A-Z]", command):
        return components
    modeled = next((c for c in components if c.kind == "task" and
                    re.search(r"\b(?:run|execut\w*|shell|command)\b", c.description, re.I)), None)
    request = replace(modeled, description=text.strip(), source_span=text.strip(),
        requested_tool="shell.run_command", requested_arguments=(("cmd", command),),
        capability_requirements=("shell.run_command",)) if modeled else RequestComponent(
            "shell", text.strip(), user_value=0.6, effort=0.12, uncertainty=0.1,
            source_span=text.strip(), requested_tool="shell.run_command",
            requested_arguments=(("cmd", command),), capability_requirements=("shell.run_command",),
            evidence_refs=("literal:shell",), confidence=0.9)
    combined = [request if c is modeled else c for c in components] if modeled else [*components, request]
    return tuple(replace(c, id=f"c{i}") for i, c in enumerate(combined[:8]))
