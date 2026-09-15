# The typed surface of ``survival.r`` is the one declared for ``survival.r_api``: that stub is
# the canonical one (python/tests/test_binding_contract.py parses it by path), so this package
# re-exports it rather than duplicating its signatures.
from ..r_api import *  # noqa: F403
