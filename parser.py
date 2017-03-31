import os
from typing import Optional, List, Tuple, Dict, Sequence, Union, Set, Callable

import syntax
from lexer import Token, TokenType, get_tokens
from abc import ABC, abstractmethod

predicate_captures_t = Dict[str, List[Union[Token, syntax.ASTNode]]]
ast_node_generator_t = Callable[[predicate_captures_t], List[syntax.ASTNode]]
class Predicate(ABC):
    def __init__(self, capture_name: str, ast_node_generator: ast_node_generator_t):
        self._capture_name = capture_name
        self._ast_node_generator = ast_node_generator
        if capture_name is None and ast_node_generator is not None:
            raise AssertionError("cannot generate a syntax node without capturing it")

    def get_capture_name(self):
        return self._capture_name

    def get_node_generator(self):
        return self._ast_node_generator

    def try_consume(self, parser: 'SyntaxParser', parent_captures: predicate_captures_t) -> Tuple[bool, List[Token], List[syntax.ASTNode]]:
        ast_node_generator = self.get_node_generator()
        capture_name = self.get_capture_name()
        if ast_node_generator is None and parent_captures is None:
            raise AssertionError("non-generating predicate must have a parent")

        with Predicate.ConsumeAttemptManager(parser) as attempt:
            # if this predicate generates a node, it must not pollute its parent's captures with its own captures
            # instead, it will remember captured tokens/nodes for itself, then generate the nodes
            # and place that in the parent's captures
            child_captures = parent_captures if not ast_node_generator else {}  # type: predicate_captures_t
            tokens = self._on_consume_attempt(parser, child_captures)
            nodes = []
            if tokens is not None:
                captures = tokens
                if ast_node_generator is not None:
                    print("generator - " + capture_name)
                    nodes = ast_node_generator(child_captures)
                    captures = nodes
                if capture_name is not None:
                    new_captures = parent_captures[capture_name] if capture_name in parent_captures else []
                    new_captures.extend(captures)
                    parent_captures[capture_name] = new_captures
            else:
                attempt.fail()

        return tokens is not None, tokens or [], nodes

    class ConsumeAttemptManager(object):
        class _FailedAttempt(Exception):
            pass

        def __init__(self, parser: 'SyntaxParser'):
            self.parser = parser
            self._saved_pos = parser.tell()
            self._success = True

        def fail(self):
            """
            Abort the matching attempt. Immediately raises an exception and exits the function.
            
            :return:
            :raise:  always
            """
            self._success = False
            raise self._FailedAttempt()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self._success:
                self.parser.seek(self._saved_pos)

            if exc_type is self._FailedAttempt:
                # supress exception
                return True

    @abstractmethod
    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: Dict[str, List[Union[Token, syntax.ASTNode]]]) -> Optional[List[Token]]:
        """
        Internal callback method used when calling `try_consume`.
        Should attempt to consume tokens from parser until the predicate is satisfied, and then return the list of tokens.
        If the matching fails, should return None.
        
        :param parser: SyntaxParser instance
        :return: list of Tokens or None
        """
        pass


class TerminalPredicate(Predicate):
    def __init__(self, token_type: TokenType, capture_name: str = None, ast_node_generator: ast_node_generator_t = None):
        super().__init__(capture_name, ast_node_generator)
        self.token_type = token_type

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        token = parser.consume(self.token_type)
        return [token] if token is not None else None


class SequencePredicate(Predicate):
    def __init__(self, *preds: Predicate, capture_name: str = None, ast_node_generator: ast_node_generator_t = None):
        super().__init__(capture_name, ast_node_generator)
        self.preds = preds

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        result = []
        for pred in self.preds:
            matched, tokens, nodes = pred.try_consume(parser, captures)
            if not matched:
                return None

            result.extend(tokens)
            
        return result
    
    
class AlternativePredicate(Predicate):
    def __init__(self, *alternatives: Predicate, capture_name: str = None, ast_node_generator: ast_node_generator_t = None):
        super().__init__(capture_name, ast_node_generator)
        self.alternatives = alternatives

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        for pred in self.alternatives:
            matched, tokens, nodes = pred.try_consume(parser, captures)
            if matched:
                return tokens

        return None


class OptionalPredicate(Predicate):
    def __init__(self, inner: Predicate, capture_name: str = None, ast_node_generator: ast_node_generator_t = None):
        super().__init__(capture_name, ast_node_generator)
        self.inner = inner

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        matched, tokens, nodes = self.inner.try_consume(parser, captures)
        return tokens or []


class RepeatingPredicate(Predicate):
    def __init__(self, inner: Predicate, optional: bool = True, capture_name: str = None, ast_node_generator: ast_node_generator_t = None):
        super().__init__(capture_name, ast_node_generator)
        self.inner = inner
        self.optional = optional

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        result = []
        matched, tokens, nodes = self.inner.try_consume(parser, captures)
        matched_any = matched
        while matched:
            result.extend(tokens)
            matched, tokens, nodes = self.inner.try_consume(parser, captures)

        if not matched_any and not self.optional:
            return None

        return result


class NamedRuleReference(Predicate):
    def __init__(self, rule_name: str, capture_name: str = None, ast_node_generator: ast_node_generator_t = None):
        super().__init__(capture_name, ast_node_generator)
        self.rule_name = rule_name
        self._real_rule = None  # type: Optional[Predicate]

    def try_consume(self, parser: 'SyntaxParser', parent_captures: predicate_captures_t):
        # TODO: store real rule, override getters, call super
        pass

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        return parser.get_named_rule(self.rule_name)._on_consume_attempt(parser, captures)


opt = OptionalPredicate
seq = SequencePredicate
alt = AlternativePredicate
many = RepeatingPredicate
ref = NamedRuleReference

def tk(token_type: Union[str, TokenType], capture_name: str = None, ast_node_generator: ast_node_generator_t = None) -> TerminalPredicate:
    if not isinstance(token_type, TokenType):
        token_type = TokenType[token_type]

    return TerminalPredicate(token_type, capture_name, ast_node_generator)

class SyntaxParser(object):
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = tokens
        self._pos = 0
        self.rules = {}  # type: Dict[str, Predicate]
        self.root_rules = set()  # type: Set[str]

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int):
        if 0 <= pos < len(self.tokens):
            self._pos = pos
        else:
            raise ValueError("Invalid position")


    def peek(self) -> Optional[Token]:
        """
        Return the next token, or None if there are no more tokens
        :return: next token
        """
        if self._pos < len(self.tokens):
            return self.tokens[self._pos]

        return None

    def consume(self, token_type: TokenType) -> Optional[Token]:
        """
        Try to consume a token of the given type. The parser position is advanced
            if and only if the token is consumed (matched).

        :param token_type: the type of the target token.
        :return: the consumed token, or None
        """
        if self._pos < len(self.tokens) and self.tokens[self._pos].code == token_type:
            self._pos += 1
            return self.tokens[self._pos - 1]

        return None

    def get_named_rule(self, rule_name) -> Predicate:
        return self.rules[rule_name]

    def get_next_match(self):
        matches = []
        for rule_name, pred in self.rules.items():  # type: str, Predicate
            if rule_name not in self.root_rules:
                continue

            captures = {}
            matched, tokens, nodes = pred.try_consume(self, captures)
            print(f"{rule_name} captures: {captures}")
            if matched:
                matches.append((rule_name, tokens))

        return matches

    def add_root_rule(self, rule_name: str, pred: Predicate):
        self.add_named_rule(rule_name, pred)
        self.root_rules.add(rule_name)

    def add_named_rule(self, rule_name: str, pred: Predicate):
        self.rules[rule_name] = pred


globals().update(TokenType.__members__)

def print_captures(captures: predicate_captures_t):
    print(captures)
    return [syntax.ASTNode()]

def main():
    for test in os.listdir('tests'):
        print('=================== Analyzing file %s ==================' % test)

        # test = '4.c'
        with open(os.path.join('tests', test), 'rt') as f:
            try:
                tokens = get_tokens(f)
                parser = SyntaxParser(tokens)

                # typeBase: INT | DOUBLE | CHAR | (STRUCT ID)
                parser.add_named_rule('typeBase', alt(tk('INT'), tk('CHAR'), tk('DOUBLE'), seq(tk('STRUCT'), tk('ID'))))

                # typeName: typeBase arrayDecl?
                parser.add_named_rule('typeName', seq(ref('typeBase'), opt(ref('arrayDecl'))))

                # arrayDecl: LBRACKET expr? RBRACKET
                parser.add_named_rule('arrayDecl', seq(tk('LPAR'), tk('RPAR')))  # TODO: add `expr?`

                # declVar: typeBase ID arrayDecl? ( COMMA ID arrayDecl? )* SEMICOLON
                parser.add_named_rule('declVar', seq(
                    ref('typeBase', capture_name='base_type'), tk('ID', capture_name='names'), opt(ref('arrayDecl')),
                    many(seq(tk('COMMA'), tk('ID', capture_name='names'), opt(ref('arrayDecl')))), tk('SEMICOLON'),
                    capture_name='decl_var', ast_node_generator=print_captures
                ))

                # funcArg: typeBase ID arrayDecl?
                parser.add_named_rule('funcArg', seq(ref('typeBase'), tk('ID'), opt(ref('arrayDecl'))))


                # declStruct: STRUCT ID LACC declVar* RACC SEMICOLON
                parser.add_root_rule('declStruct', seq(
                    tk('STRUCT'), tk('ID', capture_name='name'), tk('LACC'),
                    many(ref('declVar', capture_name='members')),
                    tk('RACC'), tk('SEMICOLON'),
                    capture_name='decl_struct', ast_node_generator=print_captures
                ))

                # declFunc
                parser.add_root_rule('declFunc', seq(
                    # (typeBase MUL? | VOID) ID
                    alt(seq(ref('typeBase'), opt(tk('MUL'))), tk('VOID')), tk('ID'),
                    # LPAR (funcArg (COMMA funcArg)*)? RPAR
                    tk('LPAR'), opt(seq(ref('funcArg'), many(seq(tk('COMMA'), ref('funcArg'))))), tk('RPAR')
                    # TODO: add stmCompund
                ))

                for rule_name, tokens in parser.get_next_match():
                    print(rule_name)
                    print(' '.join(map(str, tokens)))

                # pred_void = TerminalPredicate(TokenType.VOID)
                # pred_int = TerminalPredicate(TokenType.INT)
                # pred_struct = TerminalPredicate(TokenType.STRUCT)
                # print(pred_void.try_consume(parser))
                # print(pred_int.try_consume(parser))
                # print(pred_struct.try_consume(parser))
            except ValueError as e:
                print(e)

        print('=========================================================')


if __name__ == '__main__':
    exit(main())