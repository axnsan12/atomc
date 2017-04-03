from typing import List, Union
from syntax import parser, tree
from lexer import TokenType, Token
import symbols


def _get_one(captures: parser.predicate_captures_t, capture_name: str, optional=False):
    if capture_name not in captures and optional:
        return None
    
    if capture_name not in captures:
        raise KeyError(f"{capture_name} not in captures")
    
    if len(captures[capture_name]) != 1:
        raise ValueError(f"Expected {'at most' if optional else 'exactly'} 1 value for capture {capture_name}")
    
    return captures[capture_name][0]


def _get_list_opt(captures: parser.predicate_captures_t, capture_name: str):
    return captures[capture_name] if capture_name in captures else []


def partial_variable_declaration(captures: parser.predicate_captures_t):
    name_token = _get_one(captures, 'name')
    is_array = 'is_array' in captures
    array_size_expr = _get_one(captures, 'array_size_expr', optional=True)
    return [tree.PartialVariableDeclarationNode(name_token.line, name_token.value, is_array, array_size_expr)]


def multi_variable_declaration(captures: parser.predicate_captures_t):
    base_type_tokens = captures['base_type']
    base_type = symbols.BasicType.from_tokens(base_type_tokens)
    variable_nodes = captures['variables']  # type: List[tree.PartialVariableDeclarationNode]
    return [tree.VariableDeclarationNode(base_type_tokens[0].line, base_type, variable_nodes)]


def function_argument_declaration(captures: parser.predicate_captures_t):
    base_type_tokens = captures['base_type']
    base_type = symbols.BasicType.from_tokens(base_type_tokens)
    arg_node = _get_one(captures, 'arg')  # type: tree.PartialVariableDeclarationNode
    return [tree.FunctionArgumentDeclarationNode(base_type_tokens[0].line, base_type, arg_node)]


def function_declaration(captures: parser.predicate_captures_t):
    return_type_tokens = captures['return_type']
    return_type = symbols.BasicType.from_tokens(return_type_tokens)
    is_array = 'is_array' in captures
    name = _get_one(captures, 'name').value
    argument_nodes = _get_list_opt(captures, 'args')  # type: List[tree.FunctionArgumentDeclarationNode]
    function_body = _get_one(captures, 'function_body')  # type: tree.CompoundStatementNode
    return [tree.FunctionDeclarationNode(return_type_tokens[0].line, return_type, is_array, name, argument_nodes,
                                         function_body)]


def struct_declaration(captures: parser.predicate_captures_t):
    struct_token = _get_one(captures, 'struct')
    name_token = _get_one(captures, 'name')
    member_nodes = _get_list_opt(captures, 'members')  # type: List[tree.VariableDeclarationNode]
    return [tree.StructDeclarationNode(struct_token.line, name_token.value, member_nodes)]


def constant_literal(captures: parser.predicate_captures_t):
    ct_token = _get_one(captures, 'constant')
    ct_value = ct_token.value
    if ct_token.code == TokenType.CT_INT:
        ct_type = symbols.PrimitiveType(symbols.TypeName.TB_INT)
    elif ct_token.code == TokenType.CT_REAL:
        ct_type = symbols.PrimitiveType(symbols.TypeName.TB_REAL)
    elif ct_token.code == TokenType.CT_CHAR:
        ct_type = symbols.PrimitiveType(symbols.TypeName.TB_CHAR)
    elif ct_token.code == TokenType.CT_STRING:
        ct_type = symbols.ArrayType(symbols.PrimitiveType(symbols.TypeName.TB_CHAR), len(ct_value) + 1)
    else:
        raise ValueError("Unknown constant literal type")

    return [tree.ConstantLiteralNode(ct_token.line, ct_type, ct_value)]


def variable_or_function_call(captures: parser.predicate_captures_t):
    symbol_name_token = _get_one(captures, 'symbol_name')
    if 'is_function_call' in captures:
        arg_expr_nodes = _get_list_opt(captures, 'args')  # type: List[tree.ExpressionNode]
        return [tree.FunctionCallExpressionNode(symbol_name_token.line, symbol_name_token.value, arg_expr_nodes)]
    else:
        return [tree.VariableAccessNode(symbol_name_token.line, symbol_name_token.value)]


def postfix_expression(captures: parser.predicate_captures_t):
    expr = _get_one(captures, 'expr')
    postfixes = _get_list_opt(captures, 'postfixes')

    for postfix in postfixes:
        if isinstance(postfix, tree.ExpressionNode):
            expr = tree.ArrayItemAccessNode(postfix.lineno, expr, postfix)
        elif isinstance(postfix, Token) and postfix.code == TokenType.ID:
            expr = tree.StructMemberAccessNode(postfix.line, expr, postfix.value)
        else:
            raise ValueError("Unknown postfix construct")

    return [expr]

def unary_expression(captures: parser.predicate_captures_t):
    operator_token = _get_one(captures, 'operator')
    operand_expr = _get_one(captures, 'operand')  # type: tree.ExpressionNode
    if operator_token.code == TokenType.NOT:
        node_type = tree.LogicalNegationExpressionNode
    elif operator_token.code == TokenType.SUB:
        node_type = tree.ArithmeticNegationExpressionNode
    else:
        raise ValueError(f"Unknown unary operator {operator_token}")

    return [node_type(operator_token.line, operand_expr)]


def assignment_expression(captures: parser.predicate_captures_t):
    operator_token = _get_one(captures, 'operator')
    left, right = captures['operands']  # type: tree.ExpressionNode
    if operator_token.code != TokenType.ASSIGN:
        raise ValueError(f"Unexpected binary operator {operator_token}")

    return [tree.AssignmentExpressionNode(left.lineno, left, right)]


def binary_expression_left_associative(captures: parser.predicate_captures_t):
    operator_tokens = _get_list_opt(captures, 'operator')
    operands = captures['operands']  # type: List[tree.ExpressionNode]
    if not operator_tokens:
        return operands

    operator_token = operator_tokens[0]
    for op in operator_tokens[1:]:
        if op.code != operator_token.code:
            raise ValueError("Non-homogenous operator list")

    if len(operands) != len(operator_tokens) + 1:
        raise ValueError("Not enough operands")

    if operator_token.code == TokenType.OR:
        node_type = tree.LogicalOrExpressionNode
    elif operator_token.code == TokenType.AND:
        node_type = tree.LogicalAndExpressionNode
    elif operator_token.code == TokenType.LESS:
        node_type = tree.LessThanExpressionNode
    elif operator_token.code == TokenType.LESSEQ:
        node_type = tree.LessEqualExpressionNode
    elif operator_token.code == TokenType.GREATER:
        node_type = tree.GreaterThanExpressionNode
    elif operator_token.code == TokenType.GREATEREQ:
        node_type = tree.GreaterEqualExpressionNode
    elif operator_token.code == TokenType.EQUAL:
        node_type = tree.EqualExpressionNode
    elif operator_token.code == TokenType.NOTEQ:
        node_type = tree.NotEqualExpressionNode
    elif operator_token.code == TokenType.ADD:
        node_type = tree.AdditionExpressionNode
    elif operator_token.code == TokenType.SUB:
        node_type = tree.SubtractionExpressionNode
    elif operator_token.code == TokenType.MUL:
        node_type = tree.MultiplicationExpressionNode
    elif operator_token.code == TokenType.DIV:
        node_type = tree.DivisionExpressionNode
    else:
        raise ValueError(f"Unknown binary operator {operator_token}")

    expr = node_type(operands[0].lineno, operands[0], operands[1])
    for operand in operands[2:]:
        expr = node_type(operand.lineno, expr, operand)

    return [expr]


def cast_expression(captures: parser.predicate_captures_t):
    begin_token = _get_one(captures, 'begin')
    base_type = symbols.BasicType.from_tokens(captures['base_type'])
    is_array = 'is_array' in captures
    array_size_expr = _get_one(captures, 'array_size_expr', optional=True)  # type: tree.ExpressionNode
    cast_expr = _get_one(captures, 'cast_expr')  # type: tree.ExpressionNode
    return [tree.CastExpressionNode(begin_token.line, base_type, is_array, array_size_expr, cast_expr)]

def compound_statement(captures: parser.predicate_captures_t):
    begin_token = _get_one(captures, 'begin')
    instructions = _get_list_opt(captures, 'instructions') # type: List[Union[tree.StatementNode, tree.VariableDeclarationNode]]
    return [tree.CompoundStatementNode(begin_token.line, instructions)]


def expression_statement(captures: parser.predicate_captures_t):
    expr = _get_one(captures, 'expr', optional=True)  # type: tree.ExpressionNode
    semicolon_token = _get_one(captures, 'semicolon')
    if expr:
        return [tree.ExpressionStatementNode(expr.lineno, expr)]
    else:
        return [tree.EmptyStatementNode(semicolon_token.line)]


def conditional_statement(captures: parser.predicate_captures_t):
    if_token = _get_one(captures, 'if')
    condition = _get_one(captures, 'condition')  # type: tree.ExpressionNode
    body_if = _get_one(captures, 'body_if')  # type: tree.StatementNode
    body_else = _get_one(captures, 'body_else', optional=True)
    return [tree.ConditionalStatementNode(if_token.line, condition, body_if, body_else)]


def return_statement(captures: parser.predicate_captures_t):
    return_token = _get_one(captures, 'return')
    value = _get_one(captures, 'value', optional=True)  # type: tree.ExpressionNode
    return [tree.ReturnStatementNode(return_token.line, value)]


def break_statement(captures: parser.predicate_captures_t):
    break_token = _get_one(captures, 'break')
    return [tree.BreakStatementNode(break_token.line)]


def for_statement(captures: parser.predicate_captures_t):
    for_token = _get_one(captures, 'for')
    initial = _get_one(captures, 'initial', optional=True)  # type: tree.ExpressionNode
    condition = _get_one(captures, 'condition', optional=True)  # type: tree.ExpressionNode
    increment = _get_one(captures, 'increment', optional=True)  # type: tree.ExpressionNode
    body = _get_one(captures, 'body')  # type: tree.StatementNode
    return [tree.ForStatementNode(for_token.line, initial, condition, increment, body)]


def while_statement(captures: parser.predicate_captures_t):
    while_token = _get_one(captures, 'while')
    condition = _get_one(captures, 'condition')  # type: tree.ExpressionNode
    body = _get_one(captures, 'body')  # type: tree.StatementNode
    return [tree.WhileStatementNode(while_token.line, condition, body)]


def capture_passthrough(capture_name: str):
    def _passthrough(captures: parser.predicate_captures_t):
        return captures[capture_name]

    return _passthrough