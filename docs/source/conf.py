# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from typing import List

project = "pkg_tree_modeling"
copyright = "Municipality of Rotterdam"
description = "A Python package for reconstructing 3D tree geometries and computing tree- and streetscape-level metrics from segmented point clouds."
author = "Municipality of Rotterdam"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_markdown_builder", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
