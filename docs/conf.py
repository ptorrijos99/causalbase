"""Sphinx configuration for the causalbase documentation.

Build it with::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

The example gallery under ``auto_examples/`` is executed at build time, so a
broken example fails the build rather than shipping as a broken page.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causalbase import __version__  # noqa: E402

project = "causalbase"
author = "Pablo Torrijos, José A. Gámez, José M. Puerta"
copyright = "2026, causalbase developers"  # noqa: A001
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "superpowers", "Thumbs.db", ".DS_Store"]

# -- Autodoc / numpydoc -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}
autodoc_typehints = "none"
autosummary_generate = True
# ges and GES, flop and FLOP would share one stub file on a case-insensitive
# file system.
autosummary_filename_map = {
    "causalbase.search.ges": "causalbase.search.ges_function",
    "causalbase.search.xges": "causalbase.search.xges_function",
    "causalbase.order.flop": "causalbase.order.flop_function",
}

# numpydoc would otherwise emit a second, undocumented entry for every class
# attribute and method, which autodoc has already rendered.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- Gallery ----------------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r".*\.py",
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    "show_signature": False,
    "download_all_examples": False,
    # The examples draw no figures.
    "image_scrapers": (),
}

# -- HTML -------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"causalbase {version}"
html_theme_options = {
    "github_url": "https://github.com/ptorrijos99/causalbase",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": False,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
}
html_context = {
    "default_mode": "auto",
    "github_user": "ptorrijos99",
    "github_repo": "causalbase",
    "github_version": "main",
    "doc_path": "docs",
}

myst_enable_extensions = ["colon_fence", "deflist"]
# The Markdown pages carry their own top-level heading, so let MyST turn the
# first two levels into anchors that cross-page links can target.
myst_heading_anchors = 2
