import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from helpers import setup_survival_import


@pytest.fixture(scope="session")
def survival():
    return setup_survival_import()
