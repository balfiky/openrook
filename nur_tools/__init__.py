"""Deprecated import compatibility for :mod:`openrook_tools`."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import openrook_tools as _canonical

warnings.warn(
    "nur_tools is deprecated; import openrook_tools instead.",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name

    def create_module(self, spec):
        # Both spellings must share classes, registrations, and module state.
        module = importlib.import_module(self.canonical_name)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # Import machinery assigns the alias spec before calling this method.
        # Keep reload/introspection on the real source loader and module name.
        module.__spec__ = self.canonical_spec

    def get_code(self, fullname):
        # Preserve historical `python -m` invocations as well as imports.
        spec = importlib.util.find_spec(self.canonical_name)
        return spec.loader.get_code(self.canonical_name)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("nur_tools."):
            return None
        canonical_name = fullname.replace("nur_tools", "openrook_tools", 1)
        spec = importlib.util.find_spec(canonical_name)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(canonical_name),
            is_package=spec.submodule_search_locations is not None,
        )


sys.meta_path.insert(0, _AliasFinder())
__all__ = [name for name in vars(_canonical) if not name.startswith("_")]
globals().update({name: getattr(_canonical, name) for name in __all__})


def __getattr__(name: str):
    return getattr(_canonical, name)
