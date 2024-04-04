from enum import Enum
from typing import Iterator, List

from typing_extensions import Self


# StrEnum is only available on python version >= 3.11
# Makeshift class for now, along with some useful functionality.
class StrEnum(str, Enum):
    def __iter__(self) -> Iterator[Self]:
        return iter(self)

    @classmethod
    def get_all_fields(cls) -> List[Self]:
        return [field for field in cls]

    @classmethod
    def get_all_values(cls) -> List[str]:
        return [field.value for field in cls.get_all_fields()]
