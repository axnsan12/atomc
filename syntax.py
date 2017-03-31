from typing import Iterable, Sequence, Optional, Tuple

from lexer import TokenType, Token
from collections import deque
import symbols


class LFTCSyntaxError(Exception):
    def __init__(self, error_message: str, lineno: int, expected_token_type: Sequence[TokenType]):
        self.error_message = error_message
        self.lineno = lineno
        self.expected_token_type = expected_token_type

    def __str__(self):
        return "At line %d: expected token of type (%s). Message: %s" \
               % (self.lineno, ','.join(map(str, self.expected_token_type)), self.error_message)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class SyntaxParserAttempt(object):
    class _FailedAttempt(Exception):
        pass

    def __init__(self, parser: 'SyntaxParser'):
        self.parser = parser
        self._saved_pos = deque()


    def consume(self, token_type: TokenType, syntax_error_message: str = None) -> Token:
        """
        Consume and return a token of the given type, or fail the attempt if not possible.
        If `syntax_error_message` is given, a `LFTCSyntaxError` is raised. Otherwise the attempt is aborted quietly.

        :param token_type: target token type
        :param syntax_error_message: syntax error message
        :return: the consumed token
        """
        return self.consume_one_of([token_type], syntax_error_message)

    def consume_one_of(self, token_types: Sequence[TokenType], syntax_error_message: str = None) -> Token:
        """
        Consume and return a token of one of the given types, or fail the attempt if not possible.
        In case of consume failure, if `syntax_error_message` is given, a `LFTCSyntaxError` is raised.
        Otherwise the attempt is aborted quietly.

        :param token_types: target token type alternatives
        :param syntax_error_message: syntax error message
        :return: the consumed token
        """
        if not token_types:
            raise ValueError("Empty token types sequence")

        token_type = iter(token_types)
        token = None
        try:
            while token is None:
                token = self.parser.consume(next(token_type))
        except StopIteration:
            pass

        if token is None:
            if syntax_error_message is None:
                self.fail()
            else:
                peek = self.parser.peek()
                if peek is None:
                    syntax_error_message = "unexpected end of file"
                raise LFTCSyntaxError(syntax_error_message, peek.line if peek is not None else -1, token_types)

        return token

    def checkpoint(self):
        return self

    def fail(self):
        """
        Abort the matching attempt. Immediately raises _FailedAttempt, which is supressed by __exit__,
            in effect exiting the with block.

        :return: never returns
        :raise _FailedAttempt: always
        """
        raise self._FailedAttempt()

    def __enter__(self):
        self._saved_pos.append(self.parser.tell())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self._saved_pos) == 0:
            raise RuntimeError("Context manager exited more times than it was entered.")

        if exc_type is not None:
            # rewind
            self.parser.seek(self._saved_pos.popleft())

        if exc_type is self._FailedAttempt:
            # supress exception
            return True


class ASTNode(object):
    # @classmethod
    # def parse(cls, parser: SyntaxParser):
    #     pass
    pass


def parse_base_type(attempt: SyntaxParserAttempt) -> symbols.BasicType:
    # typeBase: INT | DOUBLE | CHAR | STRUCT ID ;
    type_token = attempt.consume_one_of([TokenType.INT, TokenType.DOUBLE, TokenType.CHAR, TokenType.STRUCT])

    if type_token.code == TokenType.STRUCT:
        struct_name = attempt.consume(TokenType.ID, "missing struct name in variable declaration")
        return symbols.StructType(struct_name.value)
    elif type_token.code == TokenType.INT:
        return symbols.PrimitiveType(symbols.TypeName.TB_INT)
    elif type_token.code == TokenType.DOUBLE:
        return symbols.PrimitiveType(symbols.TypeName.TB_REAL)
    elif type_token.code == TokenType.CHAR:
        return symbols.PrimitiveType(symbols.TypeName.TB_CHAR)


def parse_array_specifier(attempt: SyntaxParserAttempt) -> Optional['Expression']:
    # arrayDecl: LBRACKET expr? RBRACKET ;
    attempt.consume(TokenType.LBRACKET)

    # PyCharm is dumb and does not recognise that the with block can swallow exceptions
    # noinspection PyUnusedLocal
    size_expr = None
    with attempt.checkpoint():
        size_expr = Expression.try_parse(attempt)

    attempt.consume(TokenType.RBRACKET)
    return size_expr


class Expression(ASTNode):
    @classmethod
    def try_parse(cls, attempt: SyntaxParserAttempt) -> Optional['Expression']:
        return Expression()


class UnaryExpression(Expression):
    pass


class BinaryExpression(Expression):
    pass

class Statement(ASTNode):
    pass


class DeclarationNode(ASTNode):
    def __init__(self, lineno: int):
        self.lineno = lineno
    pass


class StructDeclarationNode(DeclarationNode):
    @classmethod
    def parse(cls, parser: 'SyntaxParser'):
        # declStruct: STRUCT ID LACC declVar* RACC SEMICOLON ;
        with SyntaxParserAttempt(parser) as attempt:
            begin = attempt.consume(TokenType.STRUCT)
            struct_name = attempt.consume(TokenType.ID, "missing struct name in declaration")
            attempt.consume(TokenType.LACC, "missing opening accolade for struct declaration")

            members = []
            member = MultipleVariableDeclarationNode.try_parse(attempt)
            while member is not None:
                members.append(member)
                member = MultipleVariableDeclarationNode.try_parse(attempt)

            attempt.consume(TokenType.RACC, "missing closing accolade for struct declaration")
            attempt.consume(TokenType.SEMICOLON, "missing semicolon after struct declaration")
            return StructDeclarationNode(begin.line, struct_name.value, members)

    def __init__(self, lineno: int, struct_name: str, member_declarations: Sequence['MultipleVariableDeclarationNode']):
        super().__init__(lineno)
        self.struct_name = struct_name
        self.member_declarations = member_declarations


class MultipleVariableDeclarationNode(DeclarationNode):
    @classmethod
    def try_parse(cls, attempt: SyntaxParserAttempt) -> Optional['MultipleVariableDeclarationNode']:
        # declVar:  typeBase ID arrayDecl? ( COMMA ID arrayDecl? )* SEMICOLON ;
        base_type = parse_base_type(attempt)
        declarations = []

        variable_name = attempt.consume(TokenType.ID, "missing variable name in declaration")

        while variable_name is not None:
            variable_name = None
            with attempt.checkpoint():
                attempt.consume(TokenType.COMMA)
                variable_name = attempt.consume(TokenType.ID, "missing variable name in declaration")

                # PyCharm is dumb and does not recognise that the with block can swallow exceptions
                # noinspection PyUnusedLocal
                is_array, arr_size_expr = False, None
                with attempt.checkpoint():
                    arr_size_expr = parse_array_specifier(attempt)
                    is_array = True

        declarations.append(SingleVariableDeclarationNode(-1, base_type, variable_name.value, is_array, arr_size_expr))
        pass

    def __init__(self, lineno: int, declarations: Sequence['SingleVariableDeclarationNode']):
        super().__init__(lineno)
        self.declarations = declarations


class SingleVariableDeclarationNode(DeclarationNode):
    # @classmethod
    # def try_parse(cls, attempt: SyntaxParserAttempt) -> Optional['SingleVariableDeclarationNode']:
    #     # declVar:  typeBase ID arrayDecl? ( COMMA ID arrayDecl? )* SEMICOLON ;
    #     base_type = parse_base_type(attempt)
    #     variable_name = attempt.consume(TokenType.ID, "missing variable name in declaration")
    #
    #     pass

    def __init__(self, lineno: int, base_type: symbols.BasicType, variable_name: str, is_array: bool, array_size_expr: Expression):
        super().__init__(lineno)


class ArrayDeclarationNode(DeclarationNode):
    @classmethod
    def parse(cls, parser: 'SyntaxParser'):
        pass


class SyntaxTree(object):
    pass