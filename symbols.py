from enum import Enum
from abc import ABC, abstractmethod

import collections
from typing import List, Optional, Iterable

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
    STRUCT = 'struct'
    DECLARATION = 'decl'
    BUILTIN = 'builtin'


class SymbolType(ABC):
    def __init__(self, base: TypeName):
        self.type_base = base

    def __str__(self):
        return self.type_base.value

    @property
    @abstractmethod
    def sizeof(self):
        raise NotImplementedError("Abstract method.")


class BasicType(SymbolType):
    def __init__(self, base: TypeName):
        super().__init__(base)
        if type(self) == BasicType and base != TypeName.TB_VOID:
            raise AssertionError("Use either PrimitveType or StructType")

    @staticmethod
    def from_tokens(tokens: List[Token]) -> 'BasicType':
        if len(tokens) == 1 and tokens[0].code == TokenType.INT:
            return TYPE_INT
        if len(tokens) == 1 and tokens[0].code == TokenType.CHAR:
            return TYPE_CHAR
        if len(tokens) == 1 and tokens[0].code == TokenType.DOUBLE:
            return TYPE_REAL
        if len(tokens) == 1 and tokens[0].code == TokenType.VOID:
            return TYPE_VOID
        if len(tokens) == 2 and tokens[0].code == TokenType.STRUCT and tokens[1].code == TokenType.ID:
            return StructType(tokens[1].value)

        raise ValueError("Invalid tokens for BasicType")

    def __eq__(self, other: 'BasicType'):
        return self.type_base == other.type_base

    def __hash__(self):
        return hash(self.type_base)

    @property
    def sizeof(self):
        if self.type_base == TypeName.TB_VOID:
            return 0


class PrimitiveType(BasicType):
    def __init__(self, base: TypeName):
        super().__init__(base)

    @property
    def sizeof(self):
        if self.type_base == TypeName.TB_CHAR:
            return 1
        elif self.type_base == TypeName.TB_INT:
            return 4
        elif self.type_base == TypeName.TB_REAL:
            return 8


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

    @property
    def sizeof(self):
        size = 0
        for member in self.struct_symbol.members:
            size += member.type.sizeof

        return size


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

    @property
    def sizeof(self):
        return self.elem_type.sizeof * (self.size or 0)


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
        return ord
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
        self.offset = None  # type: int
        self.table = None  # type: SymbolTable

    def __str__(self):
        storage = f"{self.storage.value}+{self.offset}" if self.storage not in (StorageType.BUILTIN, StorageType.DECLARATION) else str(self.storage.value)
        return f"{{{storage}}} {self.type} {self.name}"


class VariableSymbol(Symbol):
    def __init__(self, name: str, var_type: SymbolType, storage: StorageType, lineno: int):
        super().__init__(name, var_type, storage, lineno)


class FunctionSymbol(Symbol):
    def __init__(self, name: str, ret_type: SymbolType, storage: StorageType, lineno: int, *args: VariableSymbol):
        super().__init__(name, ret_type, storage, lineno)
        self.ret_type = ret_type
        self.args = args
        self.args_size = sum(arg.type.sizeof for arg in args)


class StructSymbol(Symbol):
    def __init__(self, name: str, members: List[VariableSymbol], storage: StorageType, lineno: int):
        super().__init__(name, StructType(name), storage, lineno)
        self.members = members

    def get_member_symbol(self, member_name):
        return next((mem for mem in self.members if mem.name == member_name), None)


class BuiltinSymbol(FunctionSymbol):
    def __init__(self, name: str, ret_type: SymbolType, storage: StorageType, lineno: int, *args: VariableSymbol):
        super().__init__(name, ret_type, storage, lineno, *args)


class SymbolTable(object):
    def __init__(self, scope_name: str, storage: StorageType, outer: 'SymbolTable'=None):
        self.outer = outer
        self._children = []  # type: List[SymbolTable]
        self.scope_name = scope_name
        self.storage = storage
        self._symbols = collections.OrderedDict()
        self.scope_offset = 0
        if outer:
            outer._children.append(self)
            self.scope_name = f"{outer.scope_name}.{scope_name}"
            if outer.storage == storage == StorageType.LOCAL:
                self.scope_offset = outer.scope_offset + outer.size

    def get_symbol(self, symbol_name) -> Optional[Symbol]:
        return self._get_symbol_this(symbol_name) or self._get_symbol_outer(symbol_name)

    def _get_symbol_outer(self, symbol_name) -> Optional[Symbol]:
        return self.outer.get_symbol(symbol_name) if self.outer else None

    def _get_symbol_this(self, symbol_name) -> Optional[Symbol]:
        return self._symbols.get(symbol_name, None)

    def add_symbol(self, symbol: Symbol):
        existing = self._get_symbol_this(symbol.name)
        if existing:
            raise errors.AtomCDomainError("Attempt to redefine existing symbol {} in scope {}".format(existing, self.scope_name), symbol.lineno)

        physical_storages = (StorageType.GLOBAL, StorageType.LOCAL, StorageType.STRUCT)
        if symbol.storage == StorageType.ARG:
            last_arg = next((s for s in reversed(self._symbols.values()) if s.storage == StorageType.ARG), None)  # type: Symbol
            symbol.offset = last_arg.offset - symbol.type.sizeof if last_arg else -TYPE_INT.sizeof - symbol.type.sizeof
        elif symbol.storage in physical_storages:
            last_symbol = next((s for s in reversed(self._symbols.values()) if s.storage in physical_storages), None)  # type: Symbol
            if last_symbol and last_symbol.storage != symbol.storage:
                raise ValueError(f"Mixed storage types in symbol table - {symbol.storage} and {last_symbol.storage}")
            symbol.offset = last_symbol.offset + last_symbol.type.sizeof if last_symbol else 0

        self._symbols[symbol.name] = symbol
        symbol.table = self

    def clear_self(self):
        self._symbols = {}

    def _depth(self):
        return 0 if self.outer is None else 1 + self.outer._depth()

    @property
    def symbols(self) -> List[Symbol]:
        return list(self._symbols.values()) + sum((child.symbols for child in self._children), [])

    @property
    def size(self) -> int:
        return sum(symbol.type.sizeof for symbol in self._symbols.values())

    def __str__(self):
        tabs = "\t" * self._depth()
        self_symbols = ', '.join(map(str, self._symbols.values())) if self._symbols else "<empty>"
        child_symbols = '\n'.join(map(str, self._children))
        return tabs + f"Symbol table `{self.scope_name}`: " + self_symbols + ('\n' if child_symbols else '') + child_symbols
