import inspect
from typing import Optional

import numpy as np

from common.custom_types import AttrsConverterFunc, AttrsValidatorFunc, NpArrf64

# TODO: Tests.


class AttrsConverters:
    @classmethod
    def np_f64_converter(
        cls,
        precision: Optional[int] = None,
    ) -> AttrsConverterFunc:
        def _np_array_converter(value) -> NpArrf64:
            np_value = np.array(value, dtype=np.float64)
            if precision is not None:
                np_value = np_value.round(precision)
            return np_value

        return _np_array_converter


class AttrsValidators:
    @classmethod
    def positive_validator(
        cls,
    ) -> AttrsValidatorFunc:
        def _num_args_validator(instance, attribute, value) -> None:
            del instance, attribute  # Cleaner to do this for type checking.
            assert value > 0.0

        return _num_args_validator
