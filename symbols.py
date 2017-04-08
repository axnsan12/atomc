from enum import Enum
from abc import ABC
from typing import List, Optional

import errors
from lexer import Token, TokenType


class TypeName(Enum):
    TB_INT = 'int'
    TB_CHAR = 'char'
    TB_REAL = 'double'
    TB_VOID = 'void'
    TB_STRUCT = 'struct'
    TB_FUNC = 'func'


class StorageType(Enum):
    GLOBAL = 'global'
    LOCAL = 'local'
    ARG = 'arg'
    DECLARATION = 'decl'
    BUILTIN = 'builtin'


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

    def __eq__(self, other: 'BasicType'):
        return self.type_base == other.type_base

    def __hash__(self):
        return hash(self.type_base)


class PrimitiveType(BasicType):
    def __init__(self, base: TypeName):
        super().__init__(base)


class StructType(BasicType):
    def __init__(self, struct_name: str):
        super().__init__(TypeName.TB_STRUCT)
        self.struct_name = struct_name
        self.struct_symbol = None  # type: StructSymbol

    def __str__(self):
        return f"struct {self.struct_name}"

    def __eq__(self, other: 'StructType'):
        return self.type_base == other.type_base and self.struct_name == other.struct_name

    def __hash__(self):
        return hash((self.type_base, self.struct_name))

    def set_symbol(self, symbol_table: 'SymbolTable'):
        self.struct_symbol = symbol_table.get_symbol(self.struct_name)


class ArrayType(SymbolType):
    def __init__(self, elem_type: BasicType, size: int=None):
        super().__init__(elem_type.type_base)
        self.size = size
        self.elem_type = elem_type

    def __str__(self):
        return f"{self.elem_type}[{self.size if self.size is not None else ''}]"

    def __eq__(self, other: 'ArrayType'):
        return isinstance(other, ArrayType) and self.elem_type == other.elem_type and self.size == other.size

    def __hash__(self):
        return hash((self.type_base, self.elem_type, self.size))


TYPE_VOID = BasicType(TypeName.TB_VOID)
TYPE_INT = PrimitiveType(TypeName.TB_INT)
TYPE_REAL = PrimitiveType(TypeName.TB_REAL)
TYPE_CHAR = PrimitiveType(TypeName.TB_CHAR)


def python_type(symbol_type: SymbolType):
    if symbol_type == TYPE_INT:
        return int
    if symbol_type == TYPE_REAL:
        return float
    if symbol_type == TYPE_CHAR:
        return str
    if isinstance(symbol_type, ArrayType):
        if symbol_type.elem_type == TYPE_CHAR:
            return str

    raise ValueError(f"{symbol_type} does not have a corresponding python type")


class Symbol(object):
    def __init__(self, name: str, symbol_type: SymbolType, storage: StorageType, lineno: int):
        self.name = name
        self.type = symbol_type
        self.storage = storage
        self.lineno = lineno

    def __str__(self):
        return f"{{{self.storage.value}}} {self.type} {self.name}"


class VariableSymbol(Symbol):
    def __init__(self, name: str, var_type: SymbolType, storage: StorageType, lineno: int):
        super().__init__(name, var_type, storage, lineno)


class FunctionSymbol(Symbol):
    def __init__(self, name: str, ret_type: SymbolType, storage: StorageType, lineno: int, *args: VariableSymbol):
        super().__init__(name, ret_type, storage, lineno)
        self.ret_type = ret_type
        self.args = args

    def __str__(self):
        return f"{{{self.storage.value}}} {self.ret_type} {self.name}({', '.join(map(str, self.args))})"


class StructSymbol(Symbol):
    def __init__(self, name: str, members: List[VariableSymbol], storage: StorageType, lineno: int):
        super().__init__(name, StructType(name), storage, lineno)
        self.members = members

    def get_member_symbol(self, member_name):
        return next((mem for mem in self.members if mem.name == member_name), None)

    def __str__(self):
        return f"{{{self.storage.value}}} struct {self.name} {{ {'; '.join(map(str, self.members))} }}"

class BuiltinSymbol(FunctionSymbol):
    def __init__(self, name: str, ret_type: SymbolType, storage: StorageType, lineno: int, *args: VariableSymbol):
        super().__init__(name, ret_type, storage, lineno, *args)


class SymbolTable(object):
    def __init__(self, scope_name: str, storage: StorageType, outer: 'SymbolTable'=None):
        self.outer = outer
        self._children = []  # type: List[SymbolTable]
        self.scope_name = scope_name
        self.storage = storage
        self.symbols = {}
        if outer:
            outer._children.append(self)
            self.scope_name = f"{outer.scope_name}.{scope_name}"

    def get_symbol(self, symbol_name) -> Optional[Symbol]:
        return self._get_symbol_this(symbol_name) or self._get_symbol_outer(symbol_name)

    def _get_symbol_outer(self, symbol_name) -> Optional[Symbol]:
        return self.outer.get_symbol(symbol_name) if self.outer else None

    def _get_symbol_this(self, symbol_name) -> Optional[Symbol]:
        return self.symbols.get(symbol_name, None)

    def add_symbol(self, symbol: Symbol):
        existing = self._get_symbol_this(symbol.name)
        if existing:
            raise errors.AtomCDomainError("Attempt to redefine existing symbol {} in scope {}".format(existing, self.scope_name), symbol.lineno)

        self.symbols[symbol.name] = symbol

    def clear_self(self):
        self.symbols = {}

    def _depth(self):
        return 0 if self.outer is None else 1 + self.outer._depth()

    def __str__(self):
        tabs = "\t" * self._depth()
        self_symbols = ', '.join(map(str, self.symbols.values())) if self.symbols else "<empty>"
        child_symbols = '\n'.join(map(str, self._children))
        return tabs + f"Symbol table `{self.scope_name}`: " + self_symbols + ('\n' if child_symbols else '') + child_symbols