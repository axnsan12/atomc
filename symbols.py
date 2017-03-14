from enum import Enum
from abc import ABC
from typing import List, Optional


class TypeName(Enum):
    TB_INT = 1,
    TB_STRING = 2,
    TB_CHAR = 3,
    TB_REAL = 4,
    TB_VOID = 5,
    TB_STRUCT = 6,

class SymbolType(ABC):
    def __init__(self, base: TypeName):
        self.type_base = base


class PrimitiveType(SymbolType):
    def __init__(self, base: TypeName):
        super().__init__(base)


class StructType(SymbolType):
    def __init__(self):
        super().__init__(TypeName.TB_STRUCT)


class ArrayType(SymbolType):
    def __init__(self, elem_type: PrimitiveType, size: int=None):
        super().__init__(elem_type.type_base)
        self.size = size


class Symbol(object):
    def __init__(self, name: str, symbol_type: SymbolType):
        self.name = name
        self.type = symbol_type


class FunctionSymbol(Symbol):
    def __init__(self, name: str, symbol_type: SymbolType, ret_type: SymbolType, args: List[Symbol]):
        super().__init__(name, symbol_type)
        self.ret_type = ret_type
        self.args = args


class StructSymbol(Symbol):
    def __init__(self, name: str, symbol_type: SymbolType, members: List[Symbol]):
        super().__init__(name, symbol_type)
        self.members = members


class SymbolTable(object):
    def __init__(self, scope_name: str, outer: 'SymbolTable'=None):
        self.outer = outer
        self.scope_name = scope_name
        self.symbols = {}

    def get_symbol(self, symbol_name) -> Optional[Symbol]:
        return self._get_symbol_this(symbol_name) or self._get_symbol_outer(symbol_name)

    def _get_symbol_outer(self, symbol_name) -> Optional[Symbol]:
        return self.outer.get_symbol(symbol_name) if self.outer else None

    def _get_symbol_this(self, symbol_name) -> Optional[Symbol]:
        return self.symbols.get(symbol_name, None)

    def add_symbol(self, symbol: Symbol):
        existing = self._get_symbol_this(symbol.name)
        if existing:
            raise ValueError("Attempt to redefine existing symbol {}".format(existing))

        self.symbols[symbol.name] = symbol