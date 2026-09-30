"""The optional transformer path, checked on a runner that does not have it installed.

Trap one: a heavy optional dependency present on the build machine masks a broken import
ordering in the job that installs without it. Nothing here imports sentence-transformers, and
two tests actively make it unimportable to prove the argument validation fires first. A caller
who passes an empty model name must be told that, not told to install half a gigabyte of
package they do not need.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

from rfa.embed import transformer_embedder
from rfa.errors import ConfigError


@pytest.fixture
def no_transformers(monkeypatch):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith(("sentence_transformers", "torch")):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


@pytest.mark.parametrize("bad", ["", "   "])
def test_an_empty_model_name_is_refused_before_the_import(no_transformers, bad):
    with pytest.raises(ConfigError, match="needs a model name"):
        transformer_embedder(bad)


def test_a_genuinely_missing_extra_reports_the_install_hint(no_transformers):
    with pytest.raises(ConfigError, match="pip install"):
        transformer_embedder("all-MiniLM-L6-v2")


@pytest.mark.parametrize(
    "module",
    ["rfa.corpus", "rfa.chunking", "rfa.failures", "rfa.embed", "rfa.retrieve", "rfa.evaluate"],
)
def test_the_core_imports_without_the_optional_extra(no_transformers, module):
    """A regression here breaks every install that did not ask for the transformer path."""
    assert importlib.import_module(module) is not None


def test_the_default_embedder_needs_no_extra(no_transformers):
    from rfa.embed import TfidfEmbedder

    embedder = TfidfEmbedder().fit(["alpha beta", "gamma delta"])
    assert embedder.encode("alpha")
