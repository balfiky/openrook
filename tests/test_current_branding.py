"""Prevent retired branding from returning to current product/contributor files."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RETIRED_NAME = re.compile(
    r"\bnur\b|nur[_.-]|nūr|\bNur(?:Config|App|Client|APIError)\b", re.IGNORECASE,
)

# These records are historical evidence, not current instructions.
HISTORICAL_PREFIXES = ("docs/plans/", "docs/internal/")
HISTORICAL_FILES = {"docs/REVIEW_FIXES.md"}

# Whole-file exceptions are dedicated compatibility surfaces only.
COMPATIBILITY_FILES = {
    "docs/OPENROOK_MIGRATION.md": "The single current guide to legacy names and upgrading.",
    "nur_tools/__init__.py": "Deprecated external import facade.",
    "openrook_tools/legacy.py": "Deprecated console entry points.",
}

# Elsewhere allow exact lines, so new copy cannot hide inside a large module.
COMPATIBILITY_LINES = {
    "config/environment.py": {
        '"""OpenRook environment settings with legacy NUR_* compatibility."""',
        'legacy = f"NUR_{name}"',
    },
    "config/__init__.py": {
        "from config.loader import load_config, NurConfig, OpenRookConfig",
        '__all__ = ["load_config", "OpenRookConfig", "NurConfig"]',
    },
    "config/loader.py": {"NurConfig = OpenRookConfig"},
    "runtime/app.py": {"NurApp = OpenRookApp"},
    "interface/client.py": {
        "NurAPIError = OpenRookAPIError", "NurClient = OpenRookClient",
    },
    "runtime/storage.py": {'LEGACY_USER_DB_NAME = "nur.db"'},
    "runtime/security.py": {
        'if allow_private_env.startswith(("OPENROOK_", "NUR_")):',
    },
    "runtime/research_store.py": {
        'for name in ("openrook", "project-nur"):',
    },
    "runtime/skills.py": {'legacy_tool = normalized.pop("nur_tool", None)'},
    "evals/research_comparison.py": {'environment.pop("NUR_CONFIG_DIR", None)'},
    "interface/routes_backup.py": {
        'if not name.startswith(("openrook-backup-", "nur-backup-")) or not name.endswith(".zip"):',
        'or not filename.startswith(("openrook-backup-", "nur-backup-"))',
    },
    "interface/static/lib/storage.js": {
        '[local, "nur_user_id", "openrook_user_id"],',
        '[session, "nur_chat_id", "openrook_chat_id"],',
        '[session, "nur_api_token", "openrook_api_token"],',
        '[session, "nur_wizard_dismissed", "openrook_wizard_dismissed"],',
    },
    "openrook_tools/validate.py": {'"nur_tools/__init__.py",'},
    "pyproject.toml": {
        'nur = "openrook_tools.legacy:main"',
        'nur-setup = "openrook_tools.legacy:setup"',
        'nur-uninstall = "openrook_tools.legacy:uninstall"',
        'nur-web = "openrook_tools.legacy:web"',
        'nur-genesis-reset = "openrook_tools.legacy:genesis_reset"',
        'nur-emotional-qa = "openrook_tools.legacy:emotional_qa"',
        'nur-uat = "openrook_tools.legacy:uat"',
        'nur-validate = "openrook_tools.legacy:validate"',
        'include = ["core*", "config*", "evals*", "interface*", "pipeline*", '
        '"runtime*", "openrook_tools*", "nur_tools*"]',
    },
}


def current_files():
    paths = {
        ROOT / name for name in (
            "README.md", "PRIVACY.md", "SECURITY.md", "CONTRIBUTING.md",
            "pyproject.toml", "Makefile", "main.py",
        )
    }
    paths.update((ROOT / "docs").rglob("*.md"))
    paths.update((ROOT / "docs/diagrams").glob("*.svg"))
    paths.update(ROOT.glob("runtime_config.*.yaml"))
    for directory in (
        "interface", "core", "runtime", "pipeline", "config",
        "openrook_tools", "nur_tools", "evals", ".github/workflows",
    ):
        paths.update(
            path for path in (ROOT / directory).rglob("*")
            if path.suffix in {".py", ".md", ".js", ".html", ".css", ".yaml", ".yml"}
        )
    return sorted(
        path for path in paths
        if path.is_file()
        and path.relative_to(ROOT).as_posix() not in HISTORICAL_FILES
        and not path.relative_to(ROOT).as_posix().startswith(HISTORICAL_PREFIXES)
    )


def test_current_surfaces_only_use_retired_names_for_explicit_compatibility():
    violations = []
    used_exceptions = set()
    for path in current_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in COMPATIBILITY_FILES:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not RETIRED_NAME.search(line):
                continue
            if line.strip() in COMPATIBILITY_LINES.get(relative, set()):
                used_exceptions.add((relative, line.strip()))
            else:
                violations.append(f"{relative}:{number}: {line.strip()}")
    assert not violations, "\n".join(violations)
    declared = {
        (path, line) for path, lines in COMPATIBILITY_LINES.items() for line in lines
    }
    assert used_exceptions == declared, f"Remove unused exceptions: {declared - used_exceptions}"
