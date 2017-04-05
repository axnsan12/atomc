from syntax import parser, generators

opt = parser.opt
seq = parser.seq
alt = parser.alt
many = parser.many
ref = parser.ref
tk = parser.tk

# noinspection PyDictCreation
syntax_rules = {}

# typeBase: INT | DOUBLE | CHAR | (STRUCT ID)
syntax_rules['typeBase'] = alt(
    tk('INT'),
    tk('CHAR'),
    tk('DOUBLE'),
    seq(tk('STRUCT'), tk('ID', syntax_error='expected identifier after struct keyword'))
)

# arrayDecl: LBRACKET expr? RBRACKET
syntax_rules['arrayDecl'] = seq(
    tk('LBRACKET'),
    opt(ref('expr', capture_name='array_size_expr')),
    tk('RBRACKET', syntax_error='missing closing bracket')
)

# funcArg: typeBase ID arrayDecl?
syntax_rules['funcArg'] = seq(
    ref('typeBase', capture_name='base_type'),
    ref('declVarBase', capture_name='arg', syntax_error='expected argument declaration'),
    ast_node_generator=generators.function_argument_declaration
)

# declVarBase: ID arrayDecl?
syntax_rules['declVarBase'] = seq(
    tk('ID', capture_name='name'),
    opt(ref('arrayDecl', capture_name='is_array')),
    ast_node_generator=generators.partial_variable_declaration
)

# exprAssign: exprUnary ASSIGN exprAssign | exprOr;
syntax_rules['exprAssign'] = alt(
    seq(
        ref('exprUnary', capture_name='operands'),
        tk('ASSIGN', capture_name='operator'),
        ref('exprAssign', capture_name='operands', syntax_error='missing right operand'),
        capture_name='expr', ast_node_generator=generators.assignment_expression
    ),
    ref('exprOr', capture_name='expr'),
    ast_node_generator=generators.capture_passthrough('expr')
)

# exprUnary: ( SUB | NOT ) exprUnary | exprPostfix;
syntax_rules['exprUnary'] = alt(
    seq(
        alt(tk('SUB'), tk('NOT'), capture_name='operator'),
        ref('exprUnary', capture_name='operand'),
        capture_name='expr', ast_node_generator=generators.unary_expression
    ),
    ref('exprPostfix', capture_name='expr'),
    ast_node_generator=generators.capture_passthrough('expr')
)

# exprPostfix: exprPostfix LBRACKET expr RBRACKET | exprPostfix DOT ID | exprPrimary ;
syntax_rules['exprPostfix'] = alt(
    seq(
        ref('exprPostfix', capture_name='_removed'),
        tk('LBRACKET'),
        ref('expr', capture_name='postfixes', syntax_error='expected expression for array index'),
        tk('RBRACKET', syntax_error='missing closing bracket')
    ),
    seq(
        ref('exprPostfix', capture_name='_removed'),
        tk('DOT'),
        tk('ID', capture_name='postfixes', syntax_error='struct member access exptected identifier')
    ),
    ref('exprPrimary', capture_name='expr'),
    ast_node_generator=generators.postfix_expression
)

# exprSymbolic: ID ( LPAR ( expr ( COMMA expr )* )? RPAR )?
syntax_rules['exprSymbolic'] = seq(
    tk('ID', capture_name='symbol_name'),
    opt(seq(
        tk('LPAR', capture_name='is_function_call'),
        opt(seq(
            ref('expr', capture_name='args'),
            many(seq(
                tk('COMMA'),
                ref('expr', capture_name='args', syntax_error='illegal trailing comma')
            ), optional=True)
        )),
        tk('RPAR', syntax_error='unclosed function call argument list')
    )),
    ast_node_generator=generators.variable_or_function_call
)

# exprPrimary: exprSymbolic
#   | CT_INT
#   | CT_REAL
#   | CT_CHAR
#   | CT_STRING
#   | LPAR expr RPAR ;
syntax_rules['exprPrimary'] = alt(
    ref('exprSymbolic', capture_name='expr'),
    alt(
        tk('CT_INT', capture_name='constant'),
        tk('CT_REAL', capture_name='constant'),
        tk('CT_CHAR', capture_name='constant'),
        tk('CT_STRING', capture_name='constant'),
        capture_name='expr',
        ast_node_generator=generators.constant_literal
    ),
    seq(
        tk('LPAR'),
        ref('expr', capture_name='expr'),
        tk('RPAR', syntax_error='mismatched parentheses')
    ),
    ast_node_generator=generators.capture_passthrough('expr')
)

# exprOr: exprOr OR exprAnd | exprAnd ;
syntax_rules['exprOr'] = alt(
    seq(
        ref('exprOr', capture_name='_removed'),
        tk('OR', capture_name='operator'),
        ref('exprAnd', capture_name='operands')
    ),
    ref('exprAnd', capture_name='operands'),
    ast_node_generator=generators.binary_expression_left_associative
)

# exprAnd: exprAnd AND exprEq | exprEq ;
syntax_rules['exprAnd'] = alt(
    seq(
        ref('exprAnd', capture_name='_removed'),
        tk('AND', capture_name='operator'),
        ref('exprEq', capture_name='operands')
    ),
    ref('exprEq', capture_name='operands'),
    ast_node_generator=generators.binary_expression_left_associative
)

# exprEq: exprEq ( EQUAL | NOTEQ ) exprRel | exprRel ;
syntax_rules['exprEq'] = alt(
    seq(
        ref('exprEq', capture_name='_removed'),
        alt(tk('EQUAL'), tk('NOTEQ'), capture_name='operator'),
        ref('exprRel', capture_name='operands')
    ),
    ref('exprRel', capture_name='operands'),
    ast_node_generator=generators.binary_expression_left_associative
)

# exprRel: exprRel ( LESS | LESSEQ | GREATER | GREATEREQ ) exprAdd | exprAdd ;
syntax_rules['exprRel'] = alt(
    seq(
        ref('exprRel', capture_name='_removed'),
        alt(tk('LESS'), tk('LESSEQ'), tk('GREATER'), tk('GREATEREQ'), capture_name='operator'),
        ref('exprAdd', capture_name='operands')
    ),
    ref('exprAdd', capture_name='operands'),
    ast_node_generator=generators.binary_expression_left_associative
)

# exprAdd: exprAdd ( ADD | SUB ) exprMul | exprMul ;
syntax_rules['exprAdd'] = alt(
    seq(
        ref('exprAdd', capture_name='_removed'),
        alt(tk('ADD'), tk('SUB'), capture_name='operator'),
        ref('exprMul', capture_name='operands', syntax_error="missing right operand")
    ),
    ref('exprMul', capture_name='operands'),
    ast_node_generator=generators.binary_expression_left_associative
)

# exprMul: exprMul ( MUL | DIV ) exprCast | exprCast ;
syntax_rules['exprMul'] = alt(
    seq(
        ref('exprMul', capture_name='_removed'),
        alt(tk('MUL'), tk('DIV'), capture_name='operator'),
        ref('exprCast', capture_name='operands')
    ),
    ref('exprCast', capture_name='operands'),
    ast_node_generator=generators.binary_expression_left_associative
)

# exprCast: LPAR typeName RPAR exprCast | exprUnary ;
syntax_rules['exprCast'] = alt(
    seq(
        tk('LPAR', capture_name='begin'),
        seq(
            ref('typeBase', capture_name='base_type'),
            opt(ref('arrayDecl', capture_name='is_array'))
        ),
        tk('RPAR', syntax_error='missing closing parenthesis in cast'),
        ref('exprCast', capture_name='cast_expr'),
        capture_name='expr', ast_node_generator=generators.cast_expression
    ),
    ref('exprUnary', capture_name='expr'),
    ast_node_generator=generators.capture_passthrough('expr')
)

# expr: exprAssign;
syntax_rules['expr'] = seq(
    ref('exprAssign', capture_name='expr'),
    ast_node_generator=generators.capture_passthrough('expr')
)

# stmCond: IF LPAR expr RPAR stm ( ELSE stm )?
syntax_rules['stmCond'] = seq(
    tk('IF', capture_name='if'),
    tk('LPAR', syntax_error='`if` must be followed by `(`'),
    ref('expr', capture_name='condition', syntax_error='if statement without condition'),
    tk('RPAR', syntax_error='missing closing parenthesis for if'),
    ref('stm', capture_name='body_if', syntax_error='missing if body'),
    opt(seq(
        tk('ELSE'),
        ref('stm', capture_name='body_else', syntax_error='missing else body')
    )),
    ast_node_generator=generators.conditional_statement
)

# stmWhile: WHILE LPAR expr RPAR stm
syntax_rules['stmWhile'] = seq(
    tk('WHILE', capture_name='while'),
    tk('LPAR', syntax_error='`while` must be followed by `(`'),
    ref('expr', capture_name='condition', syntax_error='while statement without condition'),
    tk('RPAR', syntax_error='missing closing parenthesis for while'),
    ref('stm', capture_name='body', syntax_error='missing while body'),
    ast_node_generator=generators.while_statement
)

# stmFor: FOR LPAR expr? SEMICOLON expr? SEMICOLON expr? RPAR stm
syntax_rules['stmFor'] = seq(
    tk('FOR', capture_name='for'),
    tk('LPAR', syntax_error='`for` must be followed by `(`'),
    opt(ref('expr', capture_name='initial')),
    tk('SEMICOLON', syntax_error='semicolon expected'),
    opt(ref('expr', capture_name='condition')),
    tk('SEMICOLON', syntax_error='semicolon expected'),
    opt(ref('expr', capture_name='increment')),
    tk('RPAR', syntax_error='missing closing parenthesis in for statement'),
    ref('stm', capture_name='body', syntax_error='missing for body'),
    ast_node_generator=generators.for_statement
)

# stmBreak: BREAK SEMICOLON
syntax_rules['stmBreak'] = seq(
    tk('BREAK', capture_name='break'),
    tk('SEMICOLON', syntax_error='semicolon expected'),
    ast_node_generator=generators.break_statement
)

# stmReturn: RETURN expr? SEMICOLON
syntax_rules['stmReturn'] = seq(
    tk('RETURN', capture_name='return'),
    opt(ref('expr', capture_name='value')),
    tk('SEMICOLON', syntax_error='missing semicolon'),
    ast_node_generator=generators.return_statement
)

# stmExpr: expr? SEMICOLON
# - rewritten for easier error handling:
#       stmExpr: (expr SEMICOLON) | SEMICOLON
syntax_rules['stmExpr'] = alt(
    seq(
        ref('expr', capture_name='expr'),
        tk('SEMICOLON', capture_name='semicolon', syntax_error='missing semicolon')
    ),
    tk('SEMICOLON', capture_name='semicolon'),
    ast_node_generator=generators.expression_statement
)

# stmCompound: LACC ( declVar | stm )* RACC ;
syntax_rules['stmCompound'] = seq(
    tk('LACC', capture_name='begin'),
    many(alt(
        ref('declVar', capture_name='instructions'),
        ref('stm', capture_name='instructions')
    )),
    tk('RACC', syntax_error='compound statement must have closing brace'),
    ast_node_generator=generators.compound_statement
)

# stm: stmCompound | stmCond | stmWhile | stmFor | stmBreak | stmReturn | stmExpr
syntax_rules['stm'] = alt(
    ref('stmCompound', capture_name='stm'),
    ref('stmCond', capture_name='stm'),
    ref('stmWhile', capture_name='stm'),
    ref('stmFor', capture_name='stm'),
    ref('stmBreak', capture_name='stm'),
    ref('stmReturn', capture_name='stm'),
    ref('stmExpr', capture_name='stm'),
    ast_node_generator=generators.capture_passthrough('stm')
)

# declVar: typeBase  ( COMMA ID arrayDecl? )* SEMICOLON
syntax_rules['declVar'] = seq(
    ref('typeBase', capture_name='base_type'),
    ref('declVarBase', capture_name='variables', syntax_error='missing variable declaration name'),
    many(seq(
        tk('COMMA'),
        ref('declVarBase', capture_name='variables', syntax_error='expected variable declaration'),
    )),
    tk('SEMICOLON', syntax_error='variable declaration list must end with semicolon'),
    ast_node_generator=generators.multi_variable_declaration
)

# declStruct: STRUCT ID LACC declVar* RACC SEMICOLON
syntax_rules['declStruct'] = seq(
    tk('STRUCT', capture_name='struct'),
    tk('ID', capture_name='name', syntax_error='expected identifier after struct keyword'),
    tk('LACC'),
    many(ref('declVar', capture_name='members'), optional=True),
    tk('RACC', syntax_error='struct declaration without closing accolade'),
    tk('SEMICOLON', syntax_error='semicolon expected after structure declaration'),
    ast_node_generator=generators.struct_declaration
)

# declFunc: ( typeBase MUL? | VOID ) ID LPAR ( funcArg ( COMMA funcArg )* )? RPAR stmCompound
syntax_rules['declFunc'] = seq(
    alt(seq(ref('typeBase', capture_name='return_type'), opt(tk('MUL', capture_name='is_array'))), tk('VOID', capture_name='return_type')),
    tk('ID', capture_name='name', syntax_error='missing name in function declaration'),
    tk('LPAR'),
    opt(seq(
        ref('funcArg', capture_name='args'),
        many(seq(
            tk('COMMA'),
            ref('funcArg', capture_name='args', syntax_error='expected function argument after comma')),
            optional=True
        )
    )),
    tk('RPAR', syntax_error='argument list missing close parenthesis'),
    ref('stmCompound', capture_name='function_body', syntax_error='missing function body'),
    ast_node_generator=generators.function_declaration
)

# unit: ( declStruct | declFunc | declVar )* END ;
root_rule = seq(
    many(alt(
        ref('declStruct', capture_name='declarations'),
        ref('declFunc', capture_name='declarations'),
        ref('declVar', capture_name='declarations')
    ), optional=True),
    tk('END', syntax_error='expected variable, function or structure declaration'),
    ast_node_generator=generators.compilation_unit
)