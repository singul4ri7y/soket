from soket.backend import *
from soket.tensor import *
from soket.dtype import *

# Flush all subnormal/denormal numbers to zero
from soket.utils.ftz import _flush_subnormals_to_zero
_flush_subnormals_to_zero()
