from __future__ import annotations
from typing import List
from soket import Tensor


class Parameter(Tensor):
    """ Special class to denote a parameter """

def _extract_params(value: object) -> List[Parameter]:
    if isinstance(value, Tensor):
        return [ value ]
    elif isinstance(value, dict):
        params = []
        for v in value.values():
            params += _extract_params(v)

        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _extract_params(v)

        return params
    elif isinstance(value, Module):
        return value.parameters()
    else:
        return []


def _extract_modules(value: object) -> List[Module]:
    if isinstance(value, Module):
        return [ value ] + _extract_modules(value.__dict__)
    elif isinstance(value, dict):
        modules = []
        for v in value.values():
            modules += _extract_modules(v)
        
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _extract_modules(v)
        
        return modules
    else:
        return []


class Module:
    def parameters(self) -> List[Parameter]:
        """ Returns the list of all the parameters in the module as a whole. """
        return _extract_params(self.__dict__)
    
    def modules(self) -> List[Module]:
        """ Returns list of all the sub-modules including current module. """
        return _extract_modules(self)
    
    def __call__(self, *args, **kwargs):
        """ Forward propagate when you call the module. """
        return self.forward(*args, **kwargs)
