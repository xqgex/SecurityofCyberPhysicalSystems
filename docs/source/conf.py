# Configuration file for the Sphinx documentation builder.

from os.path import abspath, dirname, join
from sys import path
path.insert(0, abspath(join(dirname(__file__), '..', '..', 'src')))

author = 'XqGeX'
copyright = '2022, XqGeX'
project = 'ECE 591 (004) Fall 2022 project'
release = '1.0'

autodoc_default_options = {'members': True, 'undoc-members': True, 'private-members': True, 'ignore-module-all': True}
exclude_patterns = ['__pycache__', '_build', 'Thumbs.db', '.DS_Store']
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode']
highlight_language = 'python3'
html_static_path = ['_static']
html_theme = 'alabaster'
suppress_warnings = ['ref.python']
templates_path = ['_templates']
toc_object_entries = True
toc_object_entries_show_parents = 'all'
