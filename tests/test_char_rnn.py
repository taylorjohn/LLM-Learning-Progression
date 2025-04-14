import pytest
import sys
import os

# Add src directory parent to sys.path to allow importing the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_import_char_rnn_pytorch():
    """Tests if src/char_rnn_pytorch.py can be imported without crashing."""
    try:
        # Import from the src package with underscores
        from src import char_rnn_pytorch
        assert True # If import succeeds, pass the test
    except Exception as e:
        pytest.fail(f"Failed to import src.char_rnn_pytorch: {e}")

def test_import_char_rnn_pytorchev():
    """Tests if src/char_rnn_pytorchev.py can be imported without crashing."""
    try:
        # Import from the src package with underscores
        from src import char_rnn_pytorchev
        assert True # If import succeeds, pass the test
    except Exception as e:
        pytest.fail(f"Failed to import src.char_rnn_pytorchev: {e}")

# TODO: Add more specific tests by refactoring code in src/*.py files 
# into functions/classes that can be imported and tested individually. 