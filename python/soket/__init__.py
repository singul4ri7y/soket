from .tensor import *
from .ops.functional import log, exp, logsumexp

from .dtype import _datatypes, default_datatype, get_scalar_dtype, promote_types
locals().update(_datatypes)

# Devices
from .backend.device import default_device, cpu, gpu
