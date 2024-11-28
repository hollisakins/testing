import os, sys

project = "testmodule"
release = '0.1.0'

sys.path.insert(0, os.path.abspath("../testmodule"))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

extensions = [
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",  # core library for html generation from docstrings
    "sphinx.ext.autosummary",  # create neat summary tables
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

# templates_path = ["templates"]

master_doc = "index"

html_theme = "furo"
html_title = "Test Module"
