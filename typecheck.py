import symbols
import errors
from syntax import tree

numeric_types = [symbols.TYPE_CHAR, symbols.TYPE_INT, symbols.TYPE_REAL]

def bind_struct_type(base_type: symbols.BasicType, symbol_table: symbols.SymbolTable = None, lineno: int = None):
    if isinstance(base_type, symbols.StructType):
        struct_name = base_type.struct_name
        base_type.set_symbol(symbol_table)
        if base_type.struct_symbol is None:
            raise errors.AtomCDomainError(f"Undefined symbol {struct_name}", lineno)
        if not isinstance(base_type.struct_symbol, symbols.StructSymbol):
            raise errors.AtomCDomainError(f"{struct_name} is not a struct", lineno)


def resolve_type(base_type: symbols.BasicType, is_array: bool, array_size_expr: 'tree.ExpressionNode' = None, lineno: int = None):
    real_type = base_type
    if is_array:
        if base_type == symbols.TYPE_VOID:
            raise errors.AtomCTypeError("array of type void", lineno)

        array_size = None
        if array_size_expr:
            if not array_size_expr.is_const():
                raise errors.AtomCTypeError("array size must be a constant expression", array_size_expr.lineno)
            if array_size_expr.type != symbols.TYPE_INT:
                raise errors.AtomCTypeError("array size must be an integer expression", array_size_expr.lineno)

            array_size = array_size_expr.calculate_const()
            if array_size <= 0:
                raise errors.AtomCTypeError("array size must be strictly positive", array_size_expr.lineno)

        real_type = symbols.ArrayType(base_type, array_size)

    return real_type


def is_numeric(symbol_type: symbols.SymbolType):
    return symbol_type in numeric_types


def greatest_type(left_type: symbols.SymbolType, right_type: symbols.SymbolType):
    if left_type == symbols.TYPE_REAL or right_type == symbols.TYPE_REAL:
        return symbols.TYPE_REAL
    elif left_type == symbols.TYPE_INT or right_type == symbols.TYPE_INT:
        return symbols.TYPE_INT
    elif left_type == symbols.TYPE_CHAR and right_type == symbols.TYPE_CHAR:
        return symbols.TYPE_CHAR
    else:
        raise AssertionError("Expected numeric types")


def check_cast_explicit(from_type: symbols.SymbolType, into_type: symbols.SymbolType) -> str:
    if from_type == symbols.TYPE_VOID:
        return f"expression of type void cannot be used as a value"

    if from_type == into_type:
        return ''

    if is_numeric(into_type):
        if not is_numeric(from_type):
            return f"cannot cast non-numeric type {from_type} to numeric type {into_type}"

        return ''

    if is_numeric(from_type):
        return f"cannot cast numeric type {from_type} to non-numeric type {into_type}"

    if isinstance(from_type, symbols.ArrayType) and isinstance(into_type, symbols.ArrayType):
        if from_type.elem_type != into_type.elem_type:
            return f"cannot convert between arrays of different element types - {from_type} into {into_type}"

        if from_type.size is None and into_type.size is not None:
            return f"cannot cast array type with undefined size {from_type} into sized array type {into_type}"

        return ''

    if isinstance(from_type, symbols.StructType) and isinstance(into_type, symbols.StructType):
        return f"cannot convert between struct types - {from_type} into {into_type}"

    return f"cannot convert between array and struct - {from_type} into {into_type}"


def check_cast_implicit(from_type: symbols.SymbolType, into_type: symbols.SymbolType) -> str:
    explicit_cast_error = check_cast_explicit(from_type, into_type)
    if explicit_cast_error:
        return explicit_cast_error

    if is_numeric(into_type):
        if numeric_types.index(into_type) < numeric_types.index(from_type):
            return f"narrowing conversion from {from_type} to {into_type} cannot be done implicitly; add a cast?"

        return ''

    if isinstance(from_type, symbols.ArrayType) and isinstance(into_type, symbols.ArrayType):
        # if into_type.size is None and from_type.size is not None:
        #     return f"cannot cast sized array type {from_type} into array type with undefined size {into_type} implicitly; add a cast?"

        return ''

    return ''
