import operator
import collections
from typing import Sequence, Union, Tuple
from abc import ABCMeta, abstractmethod
import symbols
import errors


def _bind_struct_type(base_type: symbols.BasicType, symbol_table: symbols.SymbolTable=None, lineno: int=None):
    if isinstance(base_type, symbols.StructType):
        struct_name = base_type.struct_name
        base_type.set_symbol(symbol_table)
        if base_type.struct_symbol is None:
            raise errors.AtomCDomainError(f"Undefined symbol {struct_name}", lineno)
        if not isinstance(base_type.struct_symbol, symbols.StructSymbol):
            raise errors.AtomCDomainError(f"{struct_name} is not a struct", lineno)


def _resolve_type(base_type: symbols.BasicType, is_array: bool, array_size_expr: 'ExpressionNode'=None):
    real_type = base_type
    if is_array:
        array_size = None
        if array_size_expr:
            if not array_size_expr.is_const():
                raise errors.AtomCTypeError("array size must be a constant expression", array_size_expr.lineno)
            if array_size_expr.type() != symbols.TYPE_INT:
                raise errors.AtomCTypeError("array size must be an integer expression", array_size_expr.lineno)
                
            array_size = array_size_expr.calculate_const()
            if array_size <= 0:
                raise errors.AtomCTypeError("array size must be strictly positive", array_size_expr.lineno)
        
        real_type = symbols.ArrayType(base_type, array_size)

    return real_type


class ASTNode(object, metaclass=ABCMeta):
    def __init__(self, lineno: int):
        self.lineno = lineno
        self.symbol_table = None  # type: symbols.SymbolTable
        if type(self) == ASTNode:
            raise AssertionError("Do not instantiate ASTNode directly. Use an appropriate subclass.")

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def bind_symbol_table(self, symbol_table: symbols.SymbolTable):
        self.symbol_table = symbol_table
        self._on_bind_symbol_table()

    def validate(self) -> bool:
        assert self.symbol_table is not None, "validate() should be called after symbol table binding"
        children_valid = []
        for name, value in vars(self).items():
            if value is None:
                continue
            if isinstance(value, ASTNode):
                children_valid.append(value.validate())
            if isinstance(value, collections.Iterable):
                children_valid.extend([e.validate() for e in value if isinstance(e, ASTNode) and e is not None])

        self_valid = True # self._on_validate()
        return self_valid and all(children_valid)

    # @abstractmethod
    # def _on_validate(self):
    #     raise NotImplementedError("Abstract method")

    @abstractmethod
    def _on_bind_symbol_table(self):
        raise NotImplementedError("Abstract method")


class ExpressionNode(ASTNode, metaclass=ABCMeta):
    def __init__(self, lineno: int):
        super().__init__(lineno)
        if type(self) == ExpressionNode:
            raise AssertionError("Do not instantiate ExpressionNode directly. Use an appropriate subclass.")

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

    @abstractmethod
    def type(self) -> symbols.SymbolType:
        raise NotImplementedError("Abstract method")


class ConstantLiteralNode(ExpressionNode):
    def __init__(self, lineno: int, constant_type: symbols.SymbolType, constant_value: Union[int, float, str]):
        super().__init__(lineno)
        self.constant_type = constant_type
        self.constant_value = constant_value

    def __str__(self):
        if self.constant_type.type_base == symbols.TypeName.TB_CHAR:
            if isinstance(self.constant_type, symbols.ArrayType):
                return f'"{self.constant_value}"'
            else:
                return f"'{self.constant_value}'"

        return str(self.constant_value)

    def _calculate_const(self) -> Union[int, float, str]:
        return self.constant_value

    def is_const(self) -> bool:
        return True

    def type(self) -> symbols.SymbolType:
        return self.constant_type

    def _on_bind_symbol_table(self):
        pass


class VariableAccessNode(ExpressionNode):
    def __init__(self, lineno: int, symbol_name: str):
        super().__init__(lineno)
        self.symbol_name = symbol_name

    def __str__(self):
        return self.symbol_name

    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Non-constant expression")

    def is_const(self) -> bool:
        return False

    def type(self) -> symbols.SymbolType:
        return self.symbol_table.get_symbol(self.symbol_name).type

    def _on_bind_symbol_table(self):
        pass


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

    def type(self) -> symbols.SymbolType:
        function_symbol = self.symbol_table.get_symbol(self.function_name)  # type: symbols.FunctionSymbol
        return function_symbol.ret_type

    def _on_bind_symbol_table(self):
        for arg in self.args:
            arg.bind_symbol_table(self.symbol_table)


class ArrayItemAccessNode(ExpressionNode):
    def __init__(self, lineno: int, array_variable: ExpressionNode, index_expression: ExpressionNode):
        super().__init__(lineno)
        self.array_variable = array_variable
        self.index_expression = index_expression

    def __str__(self):
        return f"{self.array_variable}[{self.index_expression}]"

    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Non-constant expression")

    def is_const(self) -> bool:
        return False

    def type(self) -> symbols.SymbolType:
        array_type = self.array_variable.type()
        if isinstance(array_type, symbols.ArrayType):
            return array_type.elem_type

    def _on_bind_symbol_table(self):
        self.array_variable.bind_symbol_table(self.symbol_table)
        self.index_expression.bind_symbol_table(self.symbol_table)


class StructMemberAccessNode(ExpressionNode):
    def __init__(self, lineno: int, struct_variable: ExpressionNode, member_name: str):
        super().__init__(lineno)
        self.struct_variable = struct_variable
        self.member_name = member_name

    def __str__(self):
        return f"{self.struct_variable}.{self.member_name}"

    def _calculate_const(self) -> Union[int, float, str]:
        raise NotImplementedError("Non-constant expression")

    def is_const(self) -> bool:
        return False

    def type(self) -> symbols.SymbolType:
        struct_type = self.struct_variable.type()  # type: symbols.StructType
        struct_symbol = struct_type.struct_symbol
        member_symbol = struct_symbol.get_member_symbol(self.member_name)
        return member_symbol.type

    def _on_bind_symbol_table(self):
        self.struct_variable.bind_symbol_table(self.symbol_table)

    def _on_validate(self):
        struct_type = self.struct_variable.type()
        if isinstance(struct_type, symbols.StructType):
            struct_symbol = struct_type.struct_symbol
            member_symbol = struct_symbol.get_member_symbol(self.member_name)
            if member_symbol is None:
                raise errors.AtomCDomainError(f"struct {struct_symbol.name} has no field named {self.member_name}", self.lineno)

            return member_symbol.type
        else:
            raise errors.AtomCTypeError("Dot notation member access requires a struct as left operand", self.lineno)


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
        target_type = symbols.python_type(self.type())  # cast the value to the right type
        return target_type(self.cast_expr.calculate_const())

    def is_const(self) -> bool:
        return self.cast_expr.is_const()

    def type(self) -> symbols.SymbolType:
        return _resolve_type(self.base_type, self.is_array, self.array_size_expr)

    def _on_bind_symbol_table(self):
        _bind_struct_type(self.base_type, self.symbol_table, self.lineno)
        if self.array_size_expr:
            self.array_size_expr.bind_symbol_table(self.symbol_table)
        self.cast_expr.bind_symbol_table(self.symbol_table)


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

    def type(self) -> symbols.SymbolType:
        return symbols.TYPE_INT


class ArithmeticNegationExpressionNode(UnaryExpressionNode):
    def __init__(self, lineno: int, operand: ExpressionNode):
        super().__init__(lineno, operand)

    def __str__(self):
        return f"-{self.operand}"

    def _calculate_const(self) -> Union[int, float, str]:
        return -self.operand.calculate_const()

    def type(self) -> symbols.SymbolType:
        return self.operand.type()


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

    def type(self) -> symbols.SymbolType:
        return self.operand_right.type()


class BinaryLogicalExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode, logical_op, op_string: str):
        super().__init__(lineno, operand_left, operand_right, op_string)
        self.operator = logical_op

    def _calculate_const(self) -> Union[int, float, str]:
        left_value = self.operand_left.calculate_const()
        right_value = self.operand_right.calculate_const()
        return int(bool(self.operator(left_value, right_value)))

    def type(self) -> symbols.SymbolType:
        return symbols.TYPE_INT


class LogicalOrExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__or__, '||')


class LogicalAndExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__and__, '&&')


class LessThanExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__lt__, '<')


class LessEqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__le__, '<=')


class GreaterThanExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__gt__, '>')


class GreaterEqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__ge__, '>=')


class EqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__eq__, '==')


class NotEqualExpressionNode(BinaryLogicalExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__ne__, '!=')


class BinaryArithmeticExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode, arithmetic_op, op_string: str):
        super().__init__(lineno, operand_left, operand_right, op_string)
        self.operator = arithmetic_op

    def _calculate_const(self) -> Union[int, float, str]:
        left_value = self.operand_left.calculate_const()
        right_value = self.operand_right.calculate_const()
        target_type = symbols.python_type(self.type())
        return target_type(self.operator(left_value, right_value))

    def type(self) -> symbols.SymbolType:
        left_type = self.operand_left.type()
        right_type = self.operand_left.type()
        if left_type == symbols.TYPE_REAL or right_type == symbols.TYPE_REAL:
            return symbols.TYPE_REAL
        elif left_type == symbols.TYPE_INT or right_type == symbols.TYPE_INT:
            return symbols.TYPE_INT
        elif left_type == symbols.TYPE_CHAR and right_type == symbols.TYPE_CHAR:
            return symbols.TYPE_CHAR
        else:
            raise errors.AtomCTypeError("Arithmetic expression only support numeric operands", self.lineno)


class AdditionExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__add__, '+')


class SubtractionExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__sub__, '-')


class MultiplicationExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__mul__, '*')


class DivisionExpressionNode(BinaryArithmeticExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right, operator.__truediv__, '/')


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


class StructDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, struct_name: str, member_declarations: Sequence['VariableDeclarationNode']):
        super().__init__(lineno)
        self.struct_name = struct_name
        self.member_declarations = member_declarations

    def __str__(self):
        return f"struct {self.struct_name} {{ {' '.join(map(str, self.member_declarations))} }}"

    def get_symbols(self) -> Sequence[symbols.StructSymbol]:
        member_symbols = []
        for member in self.member_declarations:
            member_symbols.extend(member.get_symbols())

        struct_symbol = symbols.StructSymbol(self.struct_name, member_symbols, symbols.StorageType.DECLARATION)
        return [struct_symbol]

    def _on_bind_symbol_table(self):
        member_symbol_table = symbols.SymbolTable(f'struct {self.struct_name}', symbols.StorageType.DECLARATION, self.symbol_table)
        for member in self.member_declarations:
            member.bind_symbol_table(member_symbol_table)

        self.symbol_table.add_symbol(self.get_symbol())


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
            decl_type = _resolve_type(self.base_type, decl.is_array, decl.array_size_expr)
            var_symbols.append(symbols.VariableSymbol(decl.name, decl_type, self.symbol_table.storage))

        return var_symbols

    def _on_bind_symbol_table(self):
        _bind_struct_type(self.base_type, self.symbol_table, self.lineno)
        for decl in self.declarations:
            decl.bind_symbol_table(self.symbol_table)

        for symbol in self.get_symbols():
            self.symbol_table.add_symbol(symbol)


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

    def __str__(self):
        return f"{self.return_type}{'*' if self.is_array else ''} {self.name}({', '.join(map(str, self.arguments))}) {self.function_body}"

    def get_symbols(self) -> Sequence[symbols.FunctionSymbol]:
        ret_type = _resolve_type(self.return_type, self.is_array)
        arg_symbols = [arg.get_symbol() for arg in self.arguments]
        function_symbol = symbols.FunctionSymbol(self.name, ret_type, symbols.StorageType.DECLARATION, *arg_symbols)
        return [function_symbol]

    def _on_bind_symbol_table(self):
        _bind_struct_type(self.return_type, self.symbol_table, self.lineno)
        function_symbol_table = symbols.SymbolTable(f'function {self.name} locals', symbols.StorageType.LOCAL, self.symbol_table)
        for arg in self.arguments:
            arg.bind_symbol_table(function_symbol_table)

        self.function_body.disable_scoping()
        self.function_body.bind_symbol_table(function_symbol_table)
        self.symbol_table.add_symbol(self.get_symbol())


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


class WhileStatementNode(StatementNode):
    def __init__(self, lineno: int, condition: ExpressionNode, body: StatementNode=None):
        super().__init__(lineno)
        self.condition = condition
        self.body = body

    def __str__(self):
        return f"while ({self.condition})" + (f" {self.body}" if self.body is not None else ";")

    def _on_bind_symbol_table(self):
        self.condition.bind_symbol_table(self.symbol_table)
        if self.body:
            self.body.bind_symbol_table(self.symbol_table)


class ForStatementNode(StatementNode):
    def __init__(self, lineno: int, initial: ExpressionNode=None, condition: ExpressionNode=None,
                 incremnet: ExpressionNode=None, body: StatementNode=None):
        super().__init__(lineno)
        self.initial = initial
        self.condition = condition
        self.increment = incremnet
        self.body = body

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


class BreakStatementNode(StatementNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)

    def _on_bind_symbol_table(self):
        pass


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
            inner_symbol_table = symbols.SymbolTable('compound statement', symbols.StorageType.LOCAL, self.symbol_table)
        for instr in self.instructions:
            instr.bind_symbol_table(inner_symbol_table if self.scope else self.symbol_table)


class ExpressionStatementNode(StatementNode):
    def __init__(self, lineno: int, expression: ExpressionNode):
        super().__init__(lineno)
        self.expression = expression

    def __str__(self):
        return f"{self.expression};"

    def _on_bind_symbol_table(self):
        self.expression.bind_symbol_table(self.symbol_table)


class EmptyStatementNode(StatementNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)

    def __str__(self):
        return ";"

    def _on_bind_symbol_table(self):
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
