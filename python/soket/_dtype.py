from __future__ import annotations

_supported_dtypes = [ 'float16', 'float32', 'float64', 'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64', 'bool' ]

class DType:
    def __init__(self, type: str | DType = 'float32'):
        if isinstance(type, DType):
            type = type._type_str
        else:
            assert type in _supported_dtypes, f'Unsupported datatype {type}'

        self._type_str = type._type_str if isinstance(type, DType) else  \
            type

    def __str__(self) -> str:
        return self._type_str

    def __repr__(self) -> str:
        return f'soket.DType({self.__str__()})'

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other: DType | str) -> bool:
        """ Test dtype equality. """

        if isinstance(other, DType):
            return self._type_str == other._type_str
        elif isinstance(other, str):
            return self._type_str == other

        return False

    def __ne__(self, other: DType) -> bool:
        """ Test dtype inequality. """

        return not self.__eq__(other)


_datatypes = {}
for type in _supported_dtypes:
    _datatypes[type] = DType(type)

# Declare the datatypes.
locals().update(_datatypes)


## DEFAULT DATATYPE ##
default_datatype = _datatypes['float32']


# Tensor datatypes for scalars.
def get_scalar_dtype(scalar: any) -> DType:
    if isinstance(scalar, int):
        return int32
    elif isinstance(scalar, float):
        return float32
    elif isinstance(scalar, bool):
        return _datatypes['bool']

    raise ValueError(f'Unsupported scalar {scalar}')


## CUSTOM PROMOTER ##

# Promoter table excluding cases of floats and bool. Try using float as much as possible.
_PROMOTION_TABLE = {
    # Floating points
    (float16, float32): float32,
    (float16, float64): float64,
    (float32, float64): float64,

    # Unsigned + Unsigned
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,

    (uint16, uint32): uint32,
    (uint16, uint64): uint64,

    (uint32, uint64): uint64,

    # Signed + Signed
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,

    (int16, int32): int32,
    (int16, int64): int64,

    (int32, int64): int64,

    # Unsigned + Signed (use smallest signed type that can hold both)
    (uint8, int8): int16,    # max(uint8)=255, int8 max=127 → int16 needed
    (uint8, int16): int16,
    (uint8, int32): int32,
    (uint8, int64): int64,

    (uint16, int8): int32,
    (uint16, int16): int32,
    (uint16, int32): int32,
    (uint16, int64): int64,

    (uint32, int8): int64,
    (uint32, int16): int64,
    (uint32, int32): int64,
    (uint32, int64): int64,

    # Special case: uint64 + signed
    (uint64, int8): float32,
    (uint64, int16): float32,
    (uint64, int32): float32,
    (uint64, int64): float32,
}

# Construct the symmetric table
_SYMMETRIC_PROMOTION_TABLE = {}
for (a, b), result in _PROMOTION_TABLE.items():
    _SYMMETRIC_PROMOTION_TABLE[(a, b)] = result
    _SYMMETRIC_PROMOTION_TABLE[(b, a)] = result

for f in _supported_dtypes[:3]:
    for other in _supported_dtypes[3:-1]:
        _SYMMETRIC_PROMOTION_TABLE[(f, other)] = f
        _SYMMETRIC_PROMOTION_TABLE[(other, f)] = f

def promote_types(type_a: DType, type_b: DType) -> DType:
    """
    Custom datatype promoter for different datatypes used throughout
    NDArray operations.
    """

    type_a, type_b = DType(type_a), DType(type_b)

    # Nothing to do if they are equal
    if type_a == type_b:
        return type_a

    if type_a == 'bool': return type_b
    if type_b == 'bool': return type_a

    return _SYMMETRIC_PROMOTION_TABLE[(type_a, type_b)]
