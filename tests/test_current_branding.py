"""Keep every tracked path and text file on the current product name."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RETIRED_ASCII = "".join(map(chr, (110, 117, 114)))
RETIRED_ACCENTED = "".join(map(chr, (110, 363, 114)))
RETIRED_NAME = re.compile(
    rf"\b{RETIRED_ASCII}\b|{RETIRED_ASCII}[_.-]|{RETIRED_ACCENTED}"
    rf"|(?<!\\)\b{RETIRED_ASCII}(?=(?-i:[A-Z]))",
    re.IGNORECASE,
)
ALLOWED_PATH = "runtime/storage.py"
ALLOWED_LINE = f'LEGACY_USER_DB_NAME = "{RETIRED_ASCII}.db"'


def tracked_files():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    for relative in output.decode("utf-8", errors="surrogateescape").split("\0"):
        if relative and (ROOT / relative).is_file():
            yield relative, ROOT / relative


def test_entire_tracked_repository_only_keeps_the_memory_filename_exception():
    violations = []
    allowed_occurrences = 0
    for relative, path in tracked_files():
        if any(name in relative.casefold() for name in (RETIRED_ASCII, RETIRED_ACCENTED)):
            violations.append(f"{relative}: retired name in tracked path")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Compressed binary payloads are not repository text.
            continue
        for number, line in enumerate(content.splitlines(), 1):
            matches = RETIRED_NAME.findall(line)
            if not matches:
                continue
            if relative == ALLOWED_PATH and line == ALLOWED_LINE and len(matches) == 1:
                allowed_occurrences += 1
            else:
                violations.append(f"{relative}:{number}: {line.strip()}")
    assert not violations, "\n".join(violations)
    assert allowed_occurrences == 1, "Keep exactly one memory database filename exception."
