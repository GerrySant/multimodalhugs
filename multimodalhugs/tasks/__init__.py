"""
This package contains task-specific modules for multimodalhugs.
For example, the 'translation' task should be implemented in a module
named translation.py within this package.
"""

from .translation.translation_training import main as translation_training_main
from .translation.translation_generate import main as translation_generate_main