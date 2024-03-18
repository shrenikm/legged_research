from common.class_utils import StrEnum
from common.testing_utils import execute_pytest_file


def test_str_enum() -> None:
    class SE(StrEnum):
        A = "A"
        B = "B"

    it = iter(SE)
    assert next(it) == SE.A
    assert next(it) == SE.B

    assert SE.get_all_fields() == [SE.A, SE.B]
    for field in SE.get_all_fields():
        assert isinstance(field, SE)

    assert SE.get_all_values() == ["A", "B"]


if __name__ == "__main__":
    execute_pytest_file()
