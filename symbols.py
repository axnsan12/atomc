from enum import Enum
from abc import ABC
from typing import List, Optional
from lexer import Token, TokenType


class TypeName(Enum):
    TB_INT = 'int'
    TB_CHAR = 'char'
    TB_REAL = 'double'
    TB_VOID = 'void'
    TB_STRUCT = 'struct'
    TB_FUNC = 'func'

class SymbolType(ABC):
    def __init__(self, base: TypeName):
        self.type_base = base

    def __str__(self):
        return self.type_base.value


class BasicType(SymbolType):
    def __init__(self, base: TypeName):
        super().__init__(base)
        if type(self) == BasicType and base != TypeName.TB_VOID:
            raise AssertionError("Use either PrimitveType or StructType")

    @staticmethod
    def from_tokens(tokens: List[Token]) -> 'BasicType':
        if len(tokens) == 1 and tokens[0].code == TokenType.INT:
            return PrimitiveType(TypeName.TB_INT)
        if len(tokens) == 1 and tokens[0].code == TokenType.CHAR:
            return PrimitiveType(TypeName.TB_CHAR)
        if len(tokens) == 1 and tokens[0].code == TokenType.DOUBLE:
            return PrimitiveType(TypeName.TB_REAL)
        if len(tokens) == 1 and tokens[0].code == TokenType.VOID:
            return BasicType(TypeName.TB_VOID)
        if len(tokens) == 2 and tokens[0].code == TokenType.STRUCT and tokens[1].code == TokenType.ID:
            return StructType(tokens[1].value)

        raise ValueError("Invalid tokens for BasicType")


class PrimitiveType(BasicType):
    def __init__(self, base: TypeName):
        super().__init__(base)


class StructType(BasicType):
    def __init__(self, struct_name: str):
        super().__init__(TypeName.TB_STRUCT)
        self.struct_name = struct_name

    def __str__(self):
        return f"struct {self.struct_name}"

class ArrayType(SymbolType):
    def __init__(self, elem_type: BasicType, size: int=None):
        super().__init__(elem_type.type_base)
        self.size = size
        self.elem_type = elem_type

    def __str__(self):
        return f"{self.elem_type}[{self.size if self.size is not None else ''}]"


class Symbol(object):
    def __init__(self, name: str):
        self.name = name


class VariableSymbol(Symbol):
    def __init__(self, name: str, var_type: SymbolType):
        super().__init__(name)
        self.var_type = var_type


class FunctionSymbol(Symbol):
    def __init__(self, name: str, ret_type: SymbolType, args: List[Symbol]):
        super().__init__(name)
        self.ret_type = ret_type
        self.args = args


class StructSymbol(Symbol):
    def __init__(self, name: str, members: List[Symbol]):
        super().__init__(name)
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