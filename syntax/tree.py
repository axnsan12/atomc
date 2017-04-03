from typing import Sequence, Union

import symbols


class ASTNode(object):
    def __init__(self, lineno: int):
        self.lineno = lineno
        if type(self) == ASTNode:
            raise AssertionError("Do not instantiate ASTNode directly. Use an appropriate subclass.")

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class ExpressionNode(ASTNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)
        if type(self) == ExpressionNode:
            raise AssertionError("Do not instantiate ExpressionNode directly. Use an appropriate subclass.")


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



class VariableAccessNode(ExpressionNode):
    def __init__(self, lineno: int, symbol_name: str):
        super().__init__(lineno)
        self.symbol_name = symbol_name

    def __str__(self):
        return self.symbol_name


class FunctionCallExpressionNode(ExpressionNode):
    def __init__(self, lineno: int, function_name: str, args: Sequence[ExpressionNode]):
        super().__init__(lineno)
        self.function_name = function_name
        self.args = args

    def __str__(self):
        return f"{self.function_name}({', '.join(map(str, self.args))})"


class ArrayItemAccessNode(ExpressionNode):
    def __init__(self, lineno: int, array_variable: ExpressionNode, index_expression: ExpressionNode):
        super().__init__(lineno)
        self.array_variable = array_variable
        self.index_expression = index_expression

    def __str__(self):
        return f"{self.array_variable}[{self.index_expression}]"


class StructMemberAccessNode(ExpressionNode):
    def __init__(self, lineno: int, struct_variable: ExpressionNode, member_name: str):
        super().__init__(lineno)
        self.struct_variable = struct_variable
        self.member_name = member_name

    def __str__(self):
        return f"{self.struct_variable}.{self.member_name}"


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


class UnaryExpressionNode(ExpressionNode):
    def __init__(self, lineno: int, operand: ExpressionNode):
        super().__init__(lineno)
        self.operand = operand
        if type(self) == UnaryExpressionNode:
            raise AssertionError("Do not instantiate UnaryExpressionNode directly. Use an appropriate subclass.")


class LogicalNegationExpressionNode(UnaryExpressionNode):
    def __init__(self, lineno: int, operand: ExpressionNode):
        super().__init__(lineno, operand)

    def __str__(self):
        return f"!{self.operand}"


class ArithmeticNegationExpressionNode(UnaryExpressionNode):
    def __init__(self, lineno: int, operand: ExpressionNode):
        super().__init__(lineno, operand)

    def __str__(self):
        return f"-{self.operand}"


class BinaryExpressionNode(ExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno)
        self.operand_left = operand_left
        self.operand_right = operand_right
        if type(self) == BinaryExpressionNode:
            raise AssertionError("Do not instantiate BinaryExpressionNode directly. Use an appropriate subclass.")


class AssignmentExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} = {self.operand_right}"


class LogicalOrExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} || {self.operand_right}"


class LogicalAndExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} && {self.operand_right}"


class LessThanExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} < {self.operand_right}"


class LessEqualExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} <= {self.operand_right}"


class GreaterThanExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} > {self.operand_right}"


class GreaterEqualExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} >= {self.operand_right}"


class EqualExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} == {self.operand_right}"


class NotEqualExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} != {self.operand_right}"


class AdditionExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} + {self.operand_right}"


class SubtractionExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} - {self.operand_right}"


class MultiplicationExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} * {self.operand_right}"


class DivisionExpressionNode(BinaryExpressionNode):
    def __init__(self, lineno: int, operand_left: ExpressionNode, operand_right: ExpressionNode):
        super().__init__(lineno, operand_left, operand_right)

    def __str__(self):
        return f"{self.operand_left} / {self.operand_right}"


class DeclarationNode(ASTNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)
        if type(self) == DeclarationNode:
            raise AssertionError("Do not instantiate DeclarationNode directly. Use an appropriate subclass.")


class StructDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, struct_name: str, member_declarations: Sequence['VariableDeclarationNode']):
        super().__init__(lineno)
        self.struct_name = struct_name
        self.member_declarations = member_declarations

    def __str__(self):
        return f"struct {self.struct_name} {{ {' '.join(map(str, self.member_declarations))} }}"


class VariableDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, base_type: symbols.BasicType, declarations: Sequence['PartialVariableDeclarationNode']):
        super().__init__(lineno)
        self.base_type = base_type
        self.declarations = declarations

    def __str__(self):
        names = ', '.join(map(str, self.declarations))
        return f"{self.base_type} {names};"


class PartialVariableDeclarationNode(DeclarationNode):
    def __init__(self, lineno: int, name: str, is_array: bool, array_size_expr: ExpressionNode = None):
        super().__init__(lineno)
        self.name = name
        self.is_array = is_array
        self.array_size_expr = array_size_expr

    def __str__(self):
        array = f"[{self.array_size_expr if self.array_size_expr else ''}]"
        return f"{self.name}{array if self.is_array else ''}"


class FunctionArgumentDeclarationNode(VariableDeclarationNode):
    def __init__(self, lineno: int, base_type: symbols.BasicType, partial: PartialVariableDeclarationNode):
        super().__init__(lineno, base_type, [partial])

    def __str__(self):
        return super().__str__().replace(';', '')


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


class StatementNode(ASTNode):
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

class WhileStatementNode(StatementNode):
    def __init__(self, lineno: int, condition: ExpressionNode, body: StatementNode=None):
        super().__init__(lineno)
        self.condition = condition
        self.body = body

    def __str__(self):
        return f"while ({self.condition})" + (f" {self.body}" if self.body is not None else ";")


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


class BreakStatementNode(StatementNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)


class ReturnStatementNode(StatementNode):
    def __init__(self, lineno: int, value: ExpressionNode=None):
        super().__init__(lineno)
        self.value = value

    def __str__(self):
        return_value = '' if self.value is None else ' ' + str(self.value)
        return f"return{return_value};"


class CompoundStatementNode(StatementNode):
    def __init__(self, lineno: int, instructions: Sequence[Union[StatementNode, VariableDeclarationNode]]):
        super().__init__(lineno)
        self.instructions = instructions

    def __str__(self):
        return f"{{ {' '.join(map(str, self.instructions))} }}"


class ExpressionStatementNode(StatementNode):
    def __init__(self, lineno: int, expression: ExpressionNode):
        super().__init__(lineno)
        self.expression = expression

    def __str__(self):
        return f"{self.expression};"


class EmptyStatementNode(StatementNode):
    def __init__(self, lineno: int):
        super().__init__(lineno)

    def __str__(self):
        return ";"