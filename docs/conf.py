import os
import sys
import datetime
sys.path.insert(0, os.path.abspath('..'))

project = 'SerDes Validation Framework'
current_year = datetime.datetime.now().year
copyright = f'{current_year}, Mudit Bhargava'
author = 'Mudit Bhargava'
version = '1.2.0'
release = '1.2.0'

# Extensions needed for markdown support
extensions = [
    'sphinx_markdown_tables',     # Markdown tables
    'myst_parser',                # Markdown support
    'sphinx.ext.autodoc',         # API documentation
    'sphinx.ext.viewcode',        # View source code
    'sphinx.ext.napoleon',        # Google style docstrings
    'sphinx_copybutton',         # Copy button for code blocks
    'sphinx_design',             # UI components
]

# Markdown configuration
myst_enable_extensions = [
    'colon_fence',           # Alternative to code fences
    'deflist',              # Definition lists
    'dollarmath',           # Math support
    'fieldlist',            # Field lists
    'html_admonition',      # HTML admonitions
    'html_image',           # HTML images
    'linkify',              # Auto-link URLs
    'replacements',         # Text replacements
    'smartquotes',          # Smart quotes
    'strikethrough',        # Strikethrough
    'substitution',         # Substitutions
    'tasklist',            # Task lists
]
myst_heading_anchors = 6  # Enable heading anchors up to 6 levels
myst_footnote_transition = False  # Disable automatic footnote transitions
suppress_warnings = [
    "myst.xref_missing", 
    "myst.header"
]

# Templates and themes
templates_path = ['_templates']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Files to exclude
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    'README.md'
]


# Source file configurations
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Custom sidebar templates
html_sidebars = {
    '**': [
        'sidebar/scroll-start.html',
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/navigation.html',
        'sidebar/scroll-end.html',
    ]
}

# GitHub integration
html_context = {
    'display_github': True,
    'github_user': 'muditbhargava66',
    'github_repo': 'serdes-validation-framework',
    'github_version': 'main/docs/',
}