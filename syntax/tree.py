import operator
import collections
from typing import Sequence, Union, Set, List, Type
from abc import ABCMeta, abstractmethod
from runtime import instructions as asm
import symbols
import errors
import typecheck


class ASTNode(object, metaclass=ABCMeta):
    def __init__(self, lineno: int):
        self.lineno = lineno
        self.symbol_table = None  # type: symbols.SymbolTable
        self.current_function = None  # type: FunctionDeclarationNode
        self.current_loop = None  # type: Union[ForStatementNode, WhileStatementNode]
        if type(self) == ASTNode:
            raise AssertionError("Do not instantiate ASTNode directly. Use an appropriate subclass.")

        self._validation_lock = False

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def bind_symbol_table(self, symbol_table: symbols.SymbolTable):
        self.symbol_table = symbol_table
        self._on_bind_symbol_table()

    def _set_current_loop(self, current_loop: Union['ForStatementNode', 'WhileStatementNode']):
        self.current_loop = current_loop

    def _set_current_function(self, current_function: 'FunctionDeclarationNode'):
        self.current_function = current_function

    @property
    def _direct_children(self):
        for name, value in vars(self).items():
            if value is None or value is self:
                continue

            if isinstance(value, ASTNode):
                yield value
            if isinstance(value, collections.Iterable):
                yield from [e for e in value if isinstance(e, ASTNode) and e is not None]

    @property
    def _children(self):
        yield from self.__children_recursive(self, set())

    @staticmethod
    def __children_recursive(node: 'ASTNode', visited: Set['ASTNode']):
        for child in node._direct_children:
            if child in visited:
                continue

            visited.add(child)
            grandchildren = list(ASTNode.__children_recursive(child, visited))
            visited.update(grandchildren)

            yield child
            yield from grandchildren

    def validate(self) -> bool:
        assert self.symbol_table is not None, "validate() should be called after symbol table binding"
        if self._validation_lock:
            return True

        self._validation_lock = True
        children_valid = [child.validate() for child in self._direct_children]
        self._validation_lock = False
        return all(children_valid) and self._on_validate()

    @abstractmethod
    def _on_validate(self) -> bool:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def _on_bind_symbol_table(self):
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def compile(self, program: List['asm.Instruction']):
        raise NotImplementedError("Abstract method")


class ExpressionNode(ASTNode, metaclass=ABCMeta):
    def __init__(self, lineno: int):
        super().__init__(lineno)
        if type(self) == ExpressionNode:
            raise AssertionError("Do not instantiate ExpressionNode directly. Use an appropriate subclass.")

    def is_lval(self) -> bool:
        return False

    @abstractmethod
    def is_const(self) -> bool:
        raise NotImplementedError("Abstract method")

    def calculate_const(self) -> Union[int, float, str]:
        if not self.is_const():
            raise ValueError("Not a constant expression")

        return self._calculate_const()

    @abstractmethod
    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Abstract method")

    @property
    def type(self) -> symbols.SymbolType:
        return self._get_type()
    
    def _get_type(self) -> symbols.SymbolType:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        raise NotImplementedError("Abstract method")


class ConstantLiteralNode(ExpressionNode):
    def __init__(self, lineno: int, constant_type: symbols.SymbolType, constant_value: Union[int, float, str]):
        super().__init__(lineno)
        self.constant_type = constant_type
        self.constant_value = constant_value
        self.addr = None

    def __str__(self):
        if self.constant_type.type_base == symbols.TypeName.TB_CHAR:
            if isinstance(self.constant_type, symbols.ArrayType):
                return f'"{self.constant_value}"'
            else:
                return f"'{self.constant_value}'"

        return str(self.constant_value)

    def _calculate_const(self) -> Union[int, float, str]:
        return symbols.python_type(self.constant_type)(self.constant_value)

    def is_const(self) -> bool:
        return True

    def _get_type(self) -> symbols.SymbolType:
        return self.constant_type

    def _on_bind_symbol_table(self):
        pass

    def _on_validate(self) -> bool:
        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        if not as_rval:
            assert isinstance(self.constant_type, symbols.ArrayType), "expression cannot be used as lvalue"
            program.append(asm.PUSHCT(self.addr, asm.ADDR, self.lineno))
        else:
            assert not isinstance(self.constant_type, symbols.ArrayType), "expression cannot be used as rvalue"
            program.append(asm.PUSHCT(self.calculate_const(), asm.data_types[self.constant_type], self.lineno))


class VariableAccessNode(ExpressionNode):
    def __init__(self, lineno: int, symbol_name: str):
        super().__init__(lineno)
        self.symbol_name = symbol_name

    def __str__(self):
        return self.symbol_name

    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Non-constant expression")

    def is_lval(self) -> bool:
        return True

    def is_const(self) -> bool:
        return False

    def _get_type(self) -> symbols.SymbolType:
        return self.symbol_table.get_symbol(self.symbol_name).type

    def _on_bind_symbol_table(self):
        pass

    def _on_validate(self) -> bool:
        var = self.symbol_table.get_symbol(self.symbol_name)
        if var is None:
            raise errors.AtomCDomainError(f"undefined variable {self.symbol_name}", self.lineno)

        if not isinstance(var, symbols.VariableSymbol):
            raise errors.AtomCTypeError(f"symbol {var} cannot be used as a value", self.lineno)

        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        var = self.symbol_table.get_symbol(self.symbol_name)  # type: symbols.VariableSymbol
        if var.storage == symbols.StorageType.GLOBAL:
            program.append(asm.PUSHCT(var.offset, asm.ADDR, self.lineno))
        elif var.storage == symbols.StorageType.LOCAL or var.storage == symbols.StorageType.ARG:
            program.append(asm.LEAFP(var.table.scope_offset + var.offset, self.lineno))
        if as_rval:
            assert not isinstance(var.type, symbols.ArrayType), "array expression cannot be used as a rvalue"
            program.append(asm.LOAD(var.type.sizeof, self.lineno))


class FunctionCallExpressionNode(ExpressionNode):
    def __init__(self, lineno: int, function_name: str, args: Sequence[ExpressionNode]):
        super().__init__(lineno)
        self.function_name = function_name
        self.args = args

    def __str__(self):
        return f"{self.function_name}({', '.join(map(str, self.args))})"

    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Non-constant expression")

    def is_const(self) -> bool:
        return False

    def _get_type(self) -> symbols.SymbolType:
        function_symbol = self.symbol_table.get_symbol(self.function_name)  # type: symbols.FunctionSymbol
        return function_symbol.ret_type

    def _on_bind_symbol_table(self):
        for arg in self.args:
            arg.bind_symbol_table(self.symbol_table)

    def _on_validate(self) -> bool:
        function_symbol = self.symbol_table.get_symbol(self.function_name)
        if function_symbol is None:
            raise errors.AtomCDomainError(f"undefined function {self.function_name}", self.lineno)
        if not isinstance(function_symbol, symbols.FunctionSymbol):
            raise errors.AtomCDomainError(f"{self.function_name} is not a function", self.lineno)

        if len(self.args) != len(function_symbol.args):
            raise errors.AtomCTypeError(f"not enough arguments in call to function {self.function_name}", self.lineno)

        for narg, (arg_value, arg_def) in enumerate(zip(self.args, function_symbol.args)):
            implicit_cast_error = typecheck.check_cast_implicit(arg_value.type, arg_def.type)
            if implicit_cast_error:
                raise errors.AtomCTypeError(f"in call to {self.function_name} - argument {narg} type mismatch; {implicit_cast_error}", self.lineno)
        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        assert as_rval, "expression cannot be used an lvalue"
        function_symbol = self.symbol_table.get_symbol(self.function_name)  # type: symbols.FunctionSymbol
        for arg_value, formal_arg in zip(reversed(self.args), reversed(function_symbol.args)):
            pass_by_val = not isinstance(arg_value.type, symbols.ArrayType)  # only arrays are passed by addres
            arg_value.compile(program, as_rval=pass_by_val)
            if pass_by_val:
                asm.add_cast(arg_value.type, formal_arg.type, arg_value.lineno, program)

        if not isinstance(function_symbol, symbols.BuiltinSymbol):
            program.append(asm.CALL(function_symbol.offset, self.lineno))
        else:
            program.append(asm.CALLEXT(self.function_name, self.lineno))


class ArrayItemAccessNode(ExpressionNode):
    def __init__(self, lineno: int, array_variable: ExpressionNode, index_expression: ExpressionNode):
        super().__init__(lineno)
        self.array_variable = array_variable
        self.index_expression = index_expression

    def __str__(self):
        return f"{self.array_variable}[{self.index_expression}]"

    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Non-constant expression")

    def is_lval(self) -> bool:
        return True

    def is_const(self) -> bool:
        return False

    def _get_type(self) -> symbols.SymbolType:
        array_type = self.array_variable.type
        if isinstance(array_type, symbols.ArrayType):
            return array_type.elem_type

    def _on_bind_symbol_table(self):
        self.array_variable.bind_symbol_table(self.symbol_table)
        self.index_expression.bind_symbol_table(self.symbol_table)

    def _on_validate(self) -> bool:
        array_type = self.array_variable.type
        if not isinstance(array_type, symbols.ArrayType):
            raise errors.AtomCTypeError("Subscript operator [] applied to non-array expression", self.lineno)

        if typecheck.check_cast_implicit(self.index_expression.type, symbols.TYPE_INT):
            raise errors.AtomCTypeError("Array index expression must be of integer type", self.lineno)
        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        array_type = self.array_variable.type  # type: symbols.ArrayType
        elem_size = array_type.elem_type.sizeof
        self.array_variable.compile(program, as_rval=False)
        self.index_expression.compile(program)
        asm.add_cast(self.index_expression.type, symbols.TYPE_INT, self.lineno, program)
        if elem_size != 1:
            program.append(asm.PUSHCT(elem_size, asm.INT, self.lineno))
            program.append(asm.MUL(asm.INT, self.lineno))
        program.append(asm.OFFSET(self.lineno))
        if as_rval:
            program.append(asm.LOAD(elem_size, self.lineno))



class StructMemberAccessNode(ExpressionNode):
    def __init__(self, lineno: int, struct_variable: ExpressionNode, member_name: str):
        super().__init__(lineno)
        self.struct_variable = struct_variable
        self.member_name = member_name

    def __str__(self):
        return f"{self.struct_variable}.{self.member_name}"

    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Non-constant expression")

    def is_lval(self) -> bool:
        return True

    def is_const(self) -> bool:
        return False

    def _get_type(self) -> symbols.SymbolType:
        struct_type = self.struct_variable.type  # type: symbols.StructType
        struct_symbol = struct_type.struct_symbol
        member_symbol = struct_symbol.get_member_symbol(self.member_name)
        return member_symbol.type

    def _on_bind_symbol_table(self):
        self.struct_variable.bind_symbol_table(self.symbol_table)

    def _on_validate(self):
        struct_type = self.struct_variable.type
        if isinstance(struct_type, symbols.StructType):
            struct_symbol = struct_type.struct_symbol
            member_symbol = struct_symbol.get_member_symbol(self.member_name)
            if member_symbol is None:
                raise errors.AtomCDomainError(f"struct {struct_symbol.name} has no field named {self.member_name}", self.lineno)

            return member_symbol.type
        else:
            raise errors.AtomCTypeError("Dot notation member access requires a struct as left operand", self.lineno)

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        struct_type = self.struct_variable.type  # type: symbols.StructType
        struct_symbol = struct_type.struct_symbol
        member_symbol = struct_symbol.get_member_symbol(self.member_name)
        assert member_symbol.storage == symbols.StorageType.STRUCT
        offset = member_symbol.offset

        self.struct_variable.compile(program, as_rval=False)
        program.append(asm.PUSHCT(offset, asm.INT, self.lineno))
        program.append(asm.OFFSET(self.lineno))
        if as_rval:
            program.append(asm.LOAD(member_symbol.type.sizeof, self.lineno))


class CastExpressionNode(ExpressionNode):
    def __init__(self, lineno: int, base_type: symbols.BasicType, is_array: bool, array_size_expr: ExpressionNode, cast_expr: ExpressionNode):
        super().__init__(lineno)
        self.base_type = base_type
        self.is_array = is_array
        self.array_size_expr = array_size_expr
        self.cast_expr = cast_expr

    def __str__(self):
        array = f"[{self.array_size_expr if self.array_size_expr else ''}]"
        return f"({self.base_type}{array if self.is_array else ''}) {self.cast_expr}"

    def _calculate_const(self) -> Union[int, float, str]:
        target_type = symbols.python_type(self.type)  # cast the value to the right type
        return target_type(self.cast_expr.calculate_const())

    def is_const(self) -> bool:
        return self.cast_expr.is_const()

    def _get_type(self) -> symbols.SymbolType:
        return typecheck.resolve_type(self.base_type, self.is_array, self.array_size_expr, self.lineno)

    def _on_bind_symbol_table(self):
        typecheck.bind_struct_type(self.base_type, self.symbol_table, self.lineno)
        if self.array_size_expr:
            self.array_size_expr.bind_symbol_table(self.symbol_table)
        self.cast_expr.bind_symbol_table(self.symbol_table)

    def _on_validate(self):
        explicit_cast_error = typecheck.check_cast_explicit(self.cast_expr.type, self.type)
        if explicit_cast_error:
            raise errors.AtomCTypeError(explicit_cast_error, self.lineno)
        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        assert as_rval, "expression cannot be used an lvalue"
        self.cast_expr.compile(program)
        asm.add_cast(self.cast_expr.type, self.type, self.lineno, program)


class UnaryExpressionNode(ExpressionNode, metaclass=ABCMeta):
    def __init__(self, lineno: int, operand: ExpressionNode):
        super().__init__(lineno)
        self.operand = operand
        if type(self) == UnaryExpressionNode:
            raise AssertionError("Do not instantiate UnaryExpressionNode directly. Use an appropriate subclass.")

    def is_const(self) -> bool:
        return self.operand.is_const()

    def _on_bind_symbol_table(self):
        self.operand.bind_symbol_table(self.symbol_table)


class LogicalNegationExpressionNode(UnaryExpressionNode):
    def __init__(self, lineno: int, operand: ExpressionNode):
        super().__init__(lineno, operand)

    def __str__(self):
        return f"!{self.operand}"

    def _calculate_const(self) -> Union[int, float, str]:
        return int(self.operand.calculate_const() == 0)

    def _get_type(self) -> symbols.SymbolType:
        return symbols.TYPE_INT

    def _on_validate(self):
        operand_type = self.operand.type
        if not typecheck.is_numeric(operand_type) and not isinstance(operand_type, symbols.ArrayType):
            raise errors.AtomCTypeError("logical operators can only be applied to numeric and array types", self.lineno)

        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        assert as_rval, "expression cannot be used an lvalue"
        self.operand.compile(program)
        program.append(asm.NOT(asm.data_types[self.operand.type], self.lineno))


class ArithmeticNegationExpressionNode(UnaryExpressionNode):
    def __init__(self, lineno: int, operand: ExpressionNode):
        super().__init__(lineno, operand)

    def __str__(self):
        return f"-{self.operand}"

    def _calculate_const(self) -> Union[int, float, str]:
        return -self.operand.calculate_const()

    def _get_type(self) -> symbols.SymbolType:
        return self.operand.type

    def _on_validate(self):
        operand_type = self.operand.type
        if not typecheck.is_numeric(operand_type):
            raise errors.AtomCTypeError("arithmetic operators can only be applied to numeric types", self.lineno)

        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        assert as_rval, "expression cannot be used an lvalue"
        self.operand.compile(program)
        program.append(asm.NEG(asm.data_types[self.operand.type], self.lineno))


class BinaryExpressionNode(ExpressionNode, metaclass=ABCMeta):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode, op_string: str):
        super().__init__(lineno)
        self.operand_left = operand_left
        self.operand_right = operand_right
        self.op_string = op_string
        if type(self) == BinaryExpressionNode:
            raise AssertionError("Do not instantiate BinaryExpressionNode directly. Use an appropriate subclass.")

    def __str__(self):
        return f"{self.operand_left} {self.op_string} {self.operand_right}"

    def is_const(self) -> bool:
        return self.operand_left.is_const() and self.operand_right.is_const()

    def _on_bind_symbol_table(self):
        self.operand_left.bind_symbol_table(self.symbol_table)
        self.operand_right.bind_symbol_table(self.symbol_table)


class AssignmentExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, '=')

    def is_const(self) -> bool:
        return self.operand_right.is_const()

    def _calculate_const(self) -> Union[int, float, str]:
        return self.operand_right.calculate_const()

    def _get_type(self) -> symbols.SymbolType:
        return self.operand_right.type

    def _on_validate(self):
        if not self.operand_left.is_lval():
            raise errors.AtomCTypeError(f"assignment requires lvalue as left operand; {self.operand_left} is not an lvalue", self.lineno)
        implicit_cast_error = typecheck.check_cast_implicit(self.operand_right.type, self.operand_left.type)
        if implicit_cast_error:
            raise errors.AtomCTypeError(f"invalid assignment - {implicit_cast_error}", self.lineno)
        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        assert as_rval, "expression cannot be used an lvalue"
        left_type = self.operand_left.type
        right_type = self.operand_right.type
        # TODO: array copy, pointers?
        if not isinstance(left_type, symbols.ArrayType) and not isinstance(right_type, symbols.ArrayType):
            right_size = self.operand_right.type.sizeof
            self.operand_left.compile(program, as_rval=False)
            self.operand_right.compile(program)
            program.append(asm.INSERT(asm.ADDR.size + right_size, right_size, self.lineno))
            program.append(asm.STORE(right_size, self.lineno))


class BinaryLogicalExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode, logical_op, op_string: str, instruction: Type[asm.LogicalInstruction]):
        super().__init__(lineno, operand_left, operand_right, op_string)
        self.operator = logical_op
        self.instruction = instruction

    def _calculate_const(self) -> Union[int, float, str]:
        left_value = self.operand_left.calculate_const()
        right_value = self.operand_right.calculate_const()
        return int(bool(self.operator(left_value, right_value)))

    def _get_type(self) -> symbols.SymbolType:
        return symbols.TYPE_INT

    def _on_validate(self):
        left_type = self.operand_left.type
        right_type = self.operand_left.type
        if not typecheck.is_numeric(left_type):
            raise errors.AtomCTypeError("logical comparison operators can only be applied to numeric types", self.lineno)
        if not typecheck.is_numeric(right_type):
            raise errors.AtomCTypeError("logical comparison operators can only be applied to numeric types", self.lineno)

        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        assert as_rval, "expression cannot be used an lvalue"
        common_type = typecheck.greatest_type(self.operand_left.type, self.operand_right.type)
        self.operand_left.compile(program)
        asm.add_cast(self.operand_left.type, common_type, self.operand_left.lineno, program)
        self.operand_right.compile(program)
        asm.add_cast(self.operand_right.type, common_type, self.operand_right.lineno, program)
        program.append(self.instruction(asm.data_types[common_type], self.lineno))


class LogicalOrExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__or__, '||', asm.OR)


class LogicalAndExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__and__, '&&', asm.AND)


class LessThanExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__lt__, '<', asm.LESS)


class LessEqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__le__, '<=', asm.LESSEQ)


class GreaterThanExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__gt__, '>', asm.GREATER)


class GreaterEqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__ge__, '>=', asm.GREATEREQ)


class EqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__eq__, '==', asm.EQ)


class NotEqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__ne__, '!=', asm.NOTEQ)


class BinaryArithmeticExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode, arithmetic_op, op_string: str, instruction: Type[asm.ArithmeticInstruction]):
        super().__init__(lineno, operand_left, operand_right, op_string)
        self.operator = arithmetic_op
        self.instruction = instruction

    def _calculate_const(self) -> Union[int, float, str]:
        left_value = self.operand_left.calculate_const()
        right_value = self.operand_right.calculate_const()
        target_type = symbols.python_type(self.type)
        return target_type(self.operator(left_value, right_value))

    def _get_type(self) -> symbols.SymbolType:
        if not typecheck.is_numeric(self.operand_left.type) or not typecheck.is_numeric(self.operand_right.type):
            raise errors.AtomCTypeError("Arithmetic expressions only support numeric operands", self.lineno)
        return typecheck.greatest_type(self.operand_left.type, self.operand_left.type)

    def _on_validate(self):
        left_type = self.operand_left.type
        right_type = self.operand_left.type
        if not typecheck.is_numeric(left_type) or not typecheck.is_numeric(right_type):
            raise errors.AtomCTypeError("arithmetic operators can only be applied to numeric types", self.lineno)

        return True

    def compile(self, program: List['asm.Instruction'], as_rval: bool=True):
        assert as_rval, "expression cannot be used an lvalue"
        result_type = self.type
        self.operand_left.compile(program)
        asm.add_cast(self.operand_left.type, result_type, self.operand_left.lineno, program)
        self.operand_right.compile(program)
        asm.add_cast(self.operand_right.type, result_type, self.operand_right.lineno, program)
        program.append(self.instruction(asm.data_types[result_type], self.lineno))


class AdditionExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__add__, '+', asm.ADD)


class SubtractionExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__sub__, '-', asm.SUB)


class MultiplicationExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__mul__, '*', asm.MUL)


class DivisionExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__truediv__, '/', asm.DIV)
        # if typecheck.greatest_type(operand_left.type, operand_right.type) != symbols.TYPE_REAL:
        #     self.operator = operator.__floordiv__


class DeclarationNode(ASTNode, metaclass=ABCMeta):
    def __init__(self, lineno: int):
        super().__init__(lineno)
        if type(self) == DeclarationNode:
            raise AssertionError("Do not instantiate DeclarationNode directly. Use an appropriate subclass.")

    @abstractmethod
    def get_symbols(self) -> Sequence[symbols.Symbol]:
        raise NotImplementedError("Abstract method")

    def get_symbol(self) -> symbols.Symbol:
        _symbols = self.get_symbols()
        assert len(_symbols) == 1
        return _symbols[0]

    def compile(self, program: List['asm.Instruction']):
        raise AssertionError("Only function declarations can be compiled.")


class StructDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, struct_name: str, member_declarations: Sequence['VariableDeclarationNode']):
        super().__init__(lineno)
        self.struct_name = struct_name
        self.member_declarations = member_declarations
        self.member_symbol_table = None  # type: symbols.SymbolTable

    def __str__(self):
        return f"struct {self.struct_name} {{ {' '.join(map(str, self.member_declarations))} }}"

    def get_symbols(self) -> Sequence[symbols.StructSymbol]:
        member_symbols = list(self.member_symbol_table.symbols)  # type: List[symbols.VariableSymbol]
        assert all(isinstance(sym, symbols.VariableSymbol) for sym in member_symbols)
        struct_symbol = symbols.StructSymbol(self.struct_name, member_symbols, symbols.StorageType.DECLARATION, self.lineno)
        return [struct_symbol]

    def _on_bind_symbol_table(self):
        member_symbol_table = symbols.SymbolTable(f'struct-{self.struct_name}', symbols.StorageType.STRUCT, self.symbol_table)
        for member in self.member_declarations:
            member.bind_symbol_table(member_symbol_table)

        self.member_symbol_table = member_symbol_table
        self.symbol_table.add_symbol(self.get_symbol())

    def _on_validate(self):
        return True


class VariableDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, base_type: symbols.BasicType, declarations: Sequence['PartialVariableDeclarationNode']):
        super().__init__(lineno)
        self.base_type = base_type
        self.declarations = declarations

    def __str__(self):
        names = ', '.join(map(str, self.declarations))
        return f"{self.base_type} {names};"

    def get_symbols(self) -> Sequence[symbols.VariableSymbol]:
        var_symbols = []
        for decl in self.declarations:
            decl_type = typecheck.resolve_type(self.base_type, decl.is_array, decl.array_size_expr, self.lineno)
            var_symbols.append(symbols.VariableSymbol(decl.name, decl_type, self.symbol_table.storage, decl.lineno))

        return var_symbols

    def _on_bind_symbol_table(self):
        typecheck.bind_struct_type(self.base_type, self.symbol_table, self.lineno)
        for decl in self.declarations:
            decl.bind_symbol_table(self.symbol_table)

        for symbol in self.get_symbols():
            self.symbol_table.add_symbol(symbol)

    def _on_validate(self):
        if self.base_type == symbols.TYPE_VOID:
            raise errors.AtomCTypeError("Cannot declare variables of type void", self.lineno)
        return True


class PartialVariableDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, name: str, is_array: bool, array_size_expr: ExpressionNode=None):
        super().__init__(lineno)
        self.name = name
        self.is_array = is_array
        self.array_size_expr = array_size_expr

    def __str__(self):
        array = f"[{self.array_size_expr if self.array_size_expr else ''}]"
        return f"{self.name}{array if self.is_array else ''}"

    def _on_bind_symbol_table(self):
        if self.array_size_expr:
            self.array_size_expr.bind_symbol_table(self.symbol_table)

    def get_symbols(self):
        raise NotImplementedError("Cannot get symbol from partial declaration")

    def _on_validate(self):
        return True


class FunctionArgumentDeclarationNode(VariableDeclarationNode):
    def __init__(self, lineno: int, base_type: symbols.BasicType, partial: PartialVariableDeclarationNode):
        super().__init__(lineno, base_type, [partial])

    def __str__(self):
        return super().__str__().replace(';', '')

    def get_symbols(self) -> Sequence[symbols.VariableSymbol]:
        result = super().get_symbols()
        assert len(result) == 1
        result[0].storage = symbols.StorageType.ARG
        return result


class FunctionDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, return_type: symbols.BasicType, is_array: bool, name: str, arguments: Sequence[FunctionArgumentDeclarationNode], function_body: 'CompoundStatementNode'):
        super().__init__(lineno)
        self.return_type = return_type
        self.is_array = is_array
        self.name = name
        self.arguments = arguments
        self.function_body = function_body
        for child in self._children:
            child._set_current_function(self)

    def __str__(self):
        return f"{self.return_type}{'*' if self.is_array else ''} {self.name}({', '.join(map(str, self.arguments))}) {self.function_body}"

    def get_symbols(self) -> Sequence[symbols.FunctionSymbol]:
        ret_type = typecheck.resolve_type(self.return_type, self.is_array, lineno=self.lineno)
        arg_symbols = [arg.get_symbol() for arg in self.arguments]
        function_symbol = symbols.FunctionSymbol(self.name, ret_type, symbols.StorageType.DECLARATION, self.lineno, *arg_symbols)
        return [function_symbol]

    def _on_bind_symbol_table(self):
        typecheck.bind_struct_type(self.return_type, self.symbol_table, self.lineno)
        function_symbol_table = symbols.SymbolTable(f'function-{self.name}-locals', symbols.StorageType.LOCAL, self.symbol_table)
        for arg in self.arguments:
            arg.bind_symbol_table(function_symbol_table)

        self.function_body.disable_scoping()
        self.function_body.bind_symbol_table(function_symbol_table)
        self.symbol_table.add_symbol(self.get_symbol())

    @property
    def symbol(self) -> symbols.FunctionSymbol:
        symbol = self.symbol_table.get_symbol(self.name)
        assert isinstance(symbol, symbols.FunctionSymbol)
        return symbol

    def _on_validate(self):
        if self.return_type != symbols.TYPE_VOID and not any(isinstance(child, ReturnStatementNode) for child in self._children):
            raise errors.AtomCTypeError("missing return from non-void function", self.lineno)
        return True

    def compile(self, program: List['asm.Instruction']):
        self.symbol.offset = len(program)
        locals_size = sum(symbol.type.sizeof for symbol in self.symbol_table.symbols if symbol.storage == symbols.StorageType.LOCAL)
        program.append(asm.ENTER(locals_size, self.lineno))
        self.function_body.compile(program)
        has_return = any(isinstance(child, ReturnStatementNode) for child in self._children)
        if self.return_type == symbols.TYPE_VOID and not has_return:
            program.append(asm.RETFP(self.symbol.args_size, 0, self.lineno))


class StatementNode(ASTNode, metaclass=ABCMeta):
    def __init__(self, lineno: int):
        super().__init__(lineno)
        if type(self) == StatementNode:
            raise AssertionError("Do not instantiate StatementNode directly. Use an appropriate subclass.")


class ConditionalStatementNode(StatementNode):
    def __init__(self, lineno: int, condition: ExpressionNode, body_if: StatementNode=None, body_else: StatementNode=None):
        super().__init__(lineno)
        self.condition = condition
        self.body_if = body_if
        self.body_else = body_else

    def __str__(self):
        stm = f"if ({self.condition})"
        stm += f" {self.body_if}" if self.body_if is not None else ";"
        stm += f" else {self.body_else}" if self.body_else is not None else ""
        return stm

    def _on_bind_symbol_table(self):
        self.condition.bind_symbol_table(self.symbol_table)
        if self.body_if:
            self.body_if.bind_symbol_table(self.symbol_table)
        if self.body_else:
            self.body_else.bind_symbol_table(self.symbol_table)

    def _on_validate(self):
        condition_type = self.condition.type
        if not typecheck.is_numeric(condition_type):
            raise  errors.AtomCTypeError(f"expression of type {condition_type} cannot be used as a logical test", self.condition.lineno)

        return True

    def compile(self, program: List['asm.Instruction']):
        self.condition.compile(program)
        if self.body_if is not None:
            asm.add_cast(self.condition.type, symbols.TYPE_INT, self.condition.lineno, program)
            jmp_else = asm.JF(-1, self.condition.lineno)
            program.append(jmp_else)
            self.body_if.compile(program)
            if self.body_else is not None:
                jmp_end = asm.JMP(-1, self.body_if.lineno)
                program.append(jmp_end)
                jmp_else.addr = len(program)
                self.body_else.compile(program)
                jmp_end.addr = len(program)
            else:
                jmp_else.addr = len(program)
        else:
            program.append(asm.DROP(self.condition.type.sizeof, self.lineno))


class ForStatementNode(StatementNode):
    def __init__(self, lineno: int, initial: ExpressionNode=None, condition: ExpressionNode=None,
                 incremnet: ExpressionNode=None, body: StatementNode=None):
        super().__init__(lineno)
        self.initial = initial
        self.condition = condition
        self.increment = incremnet
        self.body = body
        self._breaks = []  # type: List[asm.JMP]
        for child in self._children:
            child._set_current_loop(self)

    def __str__(self):
        initial = '' if self.initial is None else str(self.initial)
        condition = '' if self.condition is None else ' ' + str(self.condition)
        increment = '' if self.increment is None else ' ' + str(self.increment)
        return f"for ({initial};{condition};{increment})" + (f" {self.body}" if self.body is not None else ";")

    def _on_bind_symbol_table(self):
        if self.initial:
            self.initial.bind_symbol_table(self.symbol_table)
        if self.condition:
            self.condition.bind_symbol_table(self.symbol_table)
        if self.increment:
            self.increment.bind_symbol_table(self.symbol_table)
        if self.body:
            self.body.bind_symbol_table(self.symbol_table)

    def _on_validate(self):
        condition_type = self.condition.type
        if not typecheck.is_numeric(condition_type):
            raise  errors.AtomCTypeError(f"expression of type {condition_type} cannot be used as a logical test", self.condition.lineno)

        return True

    def compile(self, program: List['asm.Instruction']):
        if self.initial is not None:
            self.initial.compile(program)
            program.append(asm.DROP(self.initial.type.sizeof, self.initial.lineno))

        jmp_to_condition = None
        if self.condition is not None:
            jmp_to_condition = asm.JMP(-1, self.lineno)
            program.append(jmp_to_condition)

        addr_body = len(program)
        self.body.compile(program)

        if self.increment is not None:
            self.increment.compile(program)
            program.append(asm.DROP(self.increment.type.sizeof, self.increment.lineno))

        if self.condition is not None:
            assert jmp_to_condition is not None
            jmp_to_condition.addr = len(program)
            self.condition.compile(program)
            asm.add_cast(self.condition.type, symbols.TYPE_INT, self.condition.lineno, program)
            program.append(asm.JT(addr_body, self.condition.lineno))
        else:
            program.append(asm.JMP(addr_body, self.lineno))
        addr_break = len(program)
        for brk in self._breaks:
            brk.addr = addr_break
        self._breaks.clear()


class WhileStatementNode(ForStatementNode):
    def __init__(self, lineno: int, condition: ExpressionNode, body: StatementNode=None):
        super().__init__(lineno, condition=condition, body=body)

    def __str__(self):
        return f"while ({self.condition})" + (f" {self.body}" if self.body is not None else ";")


class BreakStatementNode(StatementNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)

    def _on_bind_symbol_table(self):
        pass

    def _on_validate(self):
        if self.current_loop is None:
            raise errors.AtomCDomainError("break statement outside loop", self.lineno)

        return True

    def compile(self, program: List['asm.Instruction']):
        jmp = asm.JMP(-1, self.lineno)
        self.current_loop._breaks.append(jmp)
        program.append(jmp)


class ReturnStatementNode(StatementNode):
    def __init__(self, lineno: int, value: ExpressionNode=None):
        super().__init__(lineno)
        self.value = value

    def __str__(self):
        return_value = '' if self.value is None else ' ' + str(self.value)
        return f"return{return_value};"

    def _on_bind_symbol_table(self):
        if self.value:
            self.value.bind_symbol_table(self.symbol_table)

    def _on_validate(self):
        if self.current_function is None:
            raise errors.AtomCDomainError("return statement outside function", self.lineno)

        if self.value is None and self.current_function.return_type != symbols.TYPE_VOID:
            raise errors.AtomCTypeError("void return in non-void function", self.lineno)

        if self.value is not None:
            if self.current_function.return_type == symbols.TYPE_VOID:
                raise errors.AtomCTypeError("non-void return in void function", self.lineno)

            implicit_cast_error = typecheck.check_cast_implicit(self.value.type, self.current_function.return_type)
            if implicit_cast_error:
                raise errors.AtomCTypeError(f"bad return type - {implicit_cast_error}", self.lineno)

        return True

    def compile(self, program: List['asm.Instruction']):
        func = self.current_function.symbol
        if self.value is not None:
            self.value.compile(program)
            # the returned expression must be implicitly cast to the function's expected return type
            asm.add_cast(self.value.type, func.ret_type, self.value.lineno, program)

        program.append(asm.RETFP(func.args_size, func.ret_type.sizeof, self.lineno))


class CompoundStatementNode(StatementNode):
    def __init__(self, lineno: int, instructions: Sequence[Union[StatementNode, VariableDeclarationNode]]):
        super().__init__(lineno)
        self.instructions = instructions
        self.scope = True

    def disable_scoping(self):
        self.scope = False

    def __str__(self):
        return f"{{ {' '.join(map(str, self.instructions))} }}"

    def _on_bind_symbol_table(self):
        inner_symbol_table = None
        if self.scope:
            inner_symbol_table = symbols.SymbolTable('compound-statement', symbols.StorageType.LOCAL, self.symbol_table)
        for instr in self.instructions:
            instr.bind_symbol_table(inner_symbol_table if self.scope else self.symbol_table)

    def _on_validate(self):
        return True

    def compile(self, program: List['asm.Instruction']):
        for stm in self.instructions:
            if isinstance(stm, StatementNode):
                stm.compile(program)


class ExpressionStatementNode(StatementNode):
    def __init__(self, lineno: int, expression: ExpressionNode):
        super().__init__(lineno)
        self.expression = expression

    def __str__(self):
        return f"{self.expression};"

    def _on_bind_symbol_table(self):
        self.expression.bind_symbol_table(self.symbol_table)

    def _on_validate(self):
        return True

    def compile(self, program: List['asm.Instruction']):
        self.expression.compile(program)
        # the expression will leave its result on the stack, but it is not used, so we must make sure to remove it
        program.append(asm.DROP(self.expression.type.sizeof, self.lineno))


class EmptyStatementNode(StatementNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)

    def __str__(self):
        return ";"

    def _on_bind_symbol_table(self):
        pass

    def _on_validate(self):
        return True

    def compile(self, program: List['asm.Instruction']):
        pass


class UnitNode(ASTNode):
    def __init__(self, lineno: int, declarations: Sequence[DeclarationNode]):
        super().__init__(lineno)
        self.declarations = declarations

    def _on_bind_symbol_table(self):
        for declaration in self.declarations:
            declaration.bind_symbol_table(self.symbol_table)

    def __str__(self):
        return ' '.join(map(str, self.declarations))

    def _on_validate(self):
        main_func = self.symbol_table.get_symbol('main')
        if not main_func:
            raise errors.AtomCDomainError("undefined symbol `main`", 0)

        if not isinstance(main_func, symbols.FunctionSymbol):
            raise errors.AtomCTypeError("`main` is not a function", main_func.lineno)

        if len(main_func.args) != 0 or main_func.ret_type != symbols.TYPE_VOID:
            raise errors.AtomCTypeError("`main` should be a function taking no arguments and returning void", main_func.lineno)

        return True

    def compile(self, program: List['asm.Instruction']):
        for decl in self.declarations:
            if isinstance(decl, FunctionDeclarationNode):
                decl.compile(program)

