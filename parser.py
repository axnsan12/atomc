import os
from plistlib import Dict
from typing import Optional, List, Tuple, Iterable, Sequence, Union, Set
from lexer import Token, TokenType, get_tokens
from abc import ABC, abstractmethod


class Predicate(ABC):
    def __init__(self):
        pass

    def try_consume(self, parser: 'SyntaxParser') -> Tuple[bool, List[Token]]:
        with Predicate.ConsumeAttemptManager(parser) as attempt:
            tokens = self._on_consume_attempt(parser)
            matched = tokens is not None
            if not matched: attempt.fail()
        
        return matched, tokens or []
            

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
    def _on_consume_attempt(self, parser: 'SyntaxParser') -> Optional[List[Token]]:
        """
        Internal callback method used when calling `try_consume`.
        Should attempt to consume tokens from parser until the predicate is satisfied, and then return the list of tokens.
        If the matching fails, should return None.
        
        :param parser: SyntaxParser instance
        :return: list of Tokens or None
        """
        pass


class TerminalPredicate(Predicate):
    def __init__(self, token_type: TokenType):
        super().__init__()
        self.token_type = token_type

    def _on_consume_attempt(self, parser: 'SyntaxParser') -> Optional[List[Token]]:
        token = parser.consume(self.token_type)
        return [token] if token is not None else None


class SequencePredicate(Predicate):
    def __init__(self, *preds: Iterable[Predicate]):
        super().__init__()
        self.preds = preds

    def _on_consume_attempt(self, parser: 'SyntaxParser') -> Optional[List[Token]]:
        result = []
        for pred in self.preds:
            matched, tokens = pred.try_consume(parser)
            if not matched:
                return None

            result.extend(tokens)
            
        return result
    
    
class AlternativePredicate(Predicate):
    def __init__(self, *alternatives: Iterable[Predicate]):
        super().__init__()
        self.alternatives = alternatives

    def _on_consume_attempt(self, parser: 'SyntaxParser') -> Optional[List[Token]]:
        for pred in self.alternatives:
            matched, tokens = pred.try_consume(parser)
            if matched:
                return tokens

        return None


class OptionalPredicate(Predicate):
    def __init__(self, inner: Predicate):
        super().__init__()
        self.inner = inner

    def _on_consume_attempt(self, parser: 'SyntaxParser') -> Optional[List[Token]]:
        matched, tokens = self.inner.try_consume(parser)
        return tokens or []


class RepeatingPredicate(Predicate):
    def __init__(self, inner: Predicate, optional: bool=True):
        super().__init__()
        self.inner = inner
        self.optional = optional

    def _on_consume_attempt(self, parser: 'SyntaxParser') -> Optional[List[Token]]:
        result = []
        matched, tokens = self.inner.try_consume(parser)
        matched_any = matched
        while matched:
            result.extend(tokens)
            matched, tokens = self.inner.try_consume(parser)

        if not matched_any and not self.optional:
            return None

        return result


class NamedRuleReference(Predicate):
    def __init__(self, rule_name: str):
        super().__init__()
        self.rule_name = rule_name

    def _on_consume_attempt(self, parser: 'SyntaxParser') -> Optional[List[Token]]:
        return parser.get_named_rule(self.rule_name)._on_consume_attempt(parser)


maybe = OptionalPredicate
seq = SequencePredicate
alt = AlternativePredicate
many = RepeatingPredicate
ref = NamedRuleReference

def tk(token_type: Union[str, TokenType]) -> TerminalPredicate:
    if not isinstance(token_type, TokenType):
        token_type = TokenType[token_type]

    return TerminalPredicate(token_type)

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

    def consume(self, token_type: TokenType) -> Optional[Token]:
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

            matched, tokens = pred.try_consume(self)
            if matched:
                matches.append((rule_name, tokens))

        return matches

    def add_rule(self, rule_name: str, pred: Predicate):
        self.add_named_rule(rule_name, pred)
        self.root_rules.add(rule_name)

    def add_named_rule(self, rule_name: str, pred: Predicate):
        self.rules[rule_name] = pred


globals().update(TokenType.__members__)

def main():
    for test in os.listdir('tests'):
        print('=================== Analyzing file %s ==================' % test)

        # test = '4.c'
        with open(os.path.join('tests', test), 'rt') as f:
            try:
                tokens = get_tokens(f)
                parser = SyntaxParser(tokens)

                # INT | DOUBLE | CHAR | (STRUCT ID)
                parser.add_named_rule('typeBase', alt(tk('INT'), tk('CHAR'), tk('DOUBLE'), seq(tk('STRUCT'), tk('ID'))))

                # LBRACKET expr? RBRACKET
                parser.add_named_rule('arrayDecl', seq(tk('LPAR'), tk('RPAR')))  # TODO: add `expr?`

                # typeBase ID arrayDecl?
                parser.add_named_rule('funcArg', seq(ref('typeBase'), tk('ID'), maybe(ref('arrayDecl'))))


                parser.add_rule('declFunc', seq(
                    # (typeBase MUL? | VOID) ID
                    alt(seq(ref('typeBase'), maybe(tk('MUL'))), tk('VOID')), tk('ID'),
                    # LPAR (funcArg (COMMA funcArg)*)? RPAR
                    tk('LPAR'), maybe(seq(ref('funcArg'), many(seq(tk('COMMA'), ref('funcArg'))))), tk('RPAR'))
                    # TODO: add stmCompund
                )

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