# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Enable mock mode for documentation builds
os.environ['SVF_MOCK_MODE'] = '1'

# -- Project information -----------------------------------------------------

project = 'SerDes Validation Framework'
copyright = '2025, SerDes Validation Framework Contributors'
author = 'Mudit Bhargava'
release = '1.4.1'
version = '1.4.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/.pytest_cache',
    '**/node_modules',
    '**/venv',
    '**/__pycache__'
]

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,

    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for autodoc ----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = [
    'numpy',
    'matplotlib',
    'scipy',
    'pandas',
    'pyvisa',
    'serial',
    'usb',
    'ftdi',
]

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True

# -- Options for Napoleon extension ------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for MyST parser ------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_url_schemes = ("http", "https", "mailto")

# Configure mermaid as a directive
myst_fence_as_directive = {
    "mermaid": "mermaid"
}

# -- Options for copy button ------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Custom configuration ---------------------------------------------------

# Add custom roles
def setup(app):
    app.add_css_file('custom.css')
    
    # Add custom directives or roles here if needed
    pass

# -- API Documentation Configuration ----------------------------------------

# Automatically generate API documentation
autosummary_generate = True
autosummary_imported_members = True

# Group members by type
autodoc_member_order = 'groupwise'

# Include type hints in descriptions
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# -- Version and Release Information -----------------------------------------

# The short X.Y version
version = '1.4'
# The full version, including alpha/beta/rc tags
release = '1.4.0'

# -- Build Environment Configuration -----------------------------------------

# Set environment variables for consistent builds
os.environ.setdefault('PYTHONPATH', str(project_root))
os.environ.setdefault('SVF_MOCK_MODE', '1')

# Suppress warnings for mock imports
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sphinx')

# -- Custom Extensions ------------------------------------------------------

def skip_member(app, what, name, obj, skip, options):
    """Custom function to skip certain members during documentation generation"""
    
    # Skip private methods that start with underscore (except __init__)
    if name.startswith('_') and name != '__init__':
        return True
    
    # Skip test methods
    if name.startswith('test_'):
        return True
    
    return skip

def setup_custom(app):
    """Custom setup function"""
    app.connect('autodoc-skip-member', skip_member)

# Connect the custom setup
setup = setup_custom
