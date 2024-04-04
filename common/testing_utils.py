import inspect
from typing import Optional

import pytest


def execute_pytest_file(
    test_name: Optional[str] = None,
) -> None:
    calling_filename = inspect.stack()[1].filename
    pytest_flags = ["-s", "-v"]
    if test_name is None:
        pytest.main(pytest_flags + [calling_filename])
    else:
        pytest.main(pytest_flags + ["-k", test_name])
