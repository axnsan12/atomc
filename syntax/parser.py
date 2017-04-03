import json
import os

import collections

import itertools
from typing import Optional, List, Tuple, Dict, Sequence, Union, Set, Callable, Any
from syntax import tree
from lexer import Token, TokenType, get_tokens
from abc import ABC, abstractmethod


class AtomCSyntaxError(Exception):
    def __init__(self, error_message: str, lineno: int):
        self.error_message = error_message
        self.lineno = lineno

    def __str__(self):
        if self.lineno > 0:
            return f"Syntax error at line {self.lineno}: {self.error_message}"
        else:
            return f"Unexpected end of file - {self.error_message}"

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


predicate_captures_t = Dict[str, List[Union[Token, tree.ASTNode]]]
ast_node_generator_t = Callable[[predicate_captures_t], List[tree.ASTNode]]
class Predicate(ABC):
    tabs = ''

    def __init__(self, capture_name: str, ast_node_generator: ast_node_generator_t, syntax_error: str):
        self._capture_name = capture_name
        self._ast_node_generator = ast_node_generator
        self.syntax_error = syntax_error

    def get_capture_name(self):
        return self._capture_name

    def get_node_generator(self):
        return self._ast_node_generator

    def get_syntax_error(self):
        return self.syntax_error

    def try_consume(self, parser: 'SyntaxParser', parent_captures: predicate_captures_t) -> Tuple[bool, List[Token], List[tree.ASTNode]]:
        print(f"{Predicate.tabs}{self} enter")
        Predicate.tabs += '\t'
        ast_node_generator = self.get_node_generator()
        capture_name = self.get_capture_name()
        syntax_error = self.get_syntax_error()
        if ast_node_generator is None and parent_captures is None:
            raise AssertionError("non-generating predicate must have a parent")

        if capture_name is None:
            if ast_node_generator is not None:
                raise AssertionError("cannot generate a syntax node without capturing it")

        with Predicate.ConsumeAttemptManager(parser) as attempt:

            # child_captures = parent_captures if not ast_node_generator else {}  # type: predicate_captures_t
            child_captures = {}  # type: predicate_captures_t
            tokens = self._on_consume_attempt(parser, child_captures)
            nodes = []
            if tokens is not None:
                self_captures = tokens
                bubble_up_captures = child_captures.items()

                if ast_node_generator is not None:
                    print("generator - " + capture_name + " " + real_str(child_captures))
                    nodes = ast_node_generator(child_captures)
                    # if this predicate generates a node, it must not pollute its parent's captures with its own captures
                    # instead, it will remember captured tokens/nodes for itself, then generate the nodes
                    # and place that in the parent's captures
                    self_captures = nodes
                    bubble_up_captures = []

                if capture_name is not None:
                    bubble_up_captures = itertools.chain([(capture_name, self_captures)], bubble_up_captures)

                for name, captures in bubble_up_captures:
                    print(f"Capture bubble up: {name} {real_str(captures)}")
                    new_captures = parent_captures[name] if name in parent_captures else []
                    new_captures.extend(captures)
                    parent_captures[name] = new_captures

            else:
                if syntax_error is not None:
                    lineno = parser.peek().line if parser.peek() else -1
                    raise AtomCSyntaxError(syntax_error, lineno)
                attempt.fail()

        Predicate.tabs = Predicate.tabs[:-1]
        print(f"{Predicate.tabs}{self} exit ({'===MATCHED=== ' + ', '.join(map(real_str, tokens)) if tokens is not None else 'not matched'}) CAPTURED {real_str(parent_captures)}")
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
    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: Dict[str, List[Union[Token, tree.ASTNode]]]) -> Optional[List[Token]]:
        """
        Internal callback method used when calling `try_consume`.
        Should attempt to consume tokens from parser until the predicate is satisfied, and then return the list of tokens.
        If the matching fails, should return None.
        
        :param parser: SyntaxParser instance
        :return: list of Tokens or None
        """
        pass


class TerminalPredicate(Predicate):
    def __init__(self, token_type: Union[TokenType, str], *, capture_name: str = None, ast_node_generator: ast_node_generator_t = None, syntax_error: str = None):
        super().__init__(capture_name, ast_node_generator, syntax_error)
        self.token_type = token_type if isinstance(token_type, TokenType) else TokenType[token_type]

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        token = parser.consume(self.token_type)
        return [token] if token is not None else None

    def __str__(self):
        return f"{self.token_type.name}" + (f"{{{self.get_capture_name()}}}" if self.get_capture_name() is not None else "")

    def __repr__(self):
        return str(self)


class SequencePredicate(Predicate):
    def __init__(self, *preds: Predicate, capture_name: str = None, ast_node_generator: ast_node_generator_t = None, syntax_error: str = None):
        super().__init__(capture_name, ast_node_generator, syntax_error)
        self.preds = preds

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        result = []
        for pred in self.preds:
            matched, tokens, nodes = pred.try_consume(parser, captures)
            if not matched:
                return None

            result.extend(tokens)
            
        return result
    
    def __str__(self):
        return f"[{', '.join(map(str, self.preds))}]" + (f"{{{self.get_capture_name()}}}" if self.get_capture_name() is not None else "")

    def __repr__(self):
        return str(self)


class AlternativePredicate(Predicate):
    def __init__(self, *alternatives: Predicate, capture_name: str = None, ast_node_generator: ast_node_generator_t = None, syntax_error: str = None):
        super().__init__(capture_name, ast_node_generator, syntax_error)
        self.alternatives = alternatives

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        for pred in self.alternatives:
            matched, tokens, nodes = pred.try_consume(parser, captures)
            if matched:
                return tokens

        return None

    def __str__(self):
        return f"({' | '.join(map(str, self.alternatives))})" + (f"{{{self.get_capture_name()}}}" if self.get_capture_name() is not None else "")

    def __repr__(self):
        return str(self)


class OptionalPredicate(Predicate):
    def __init__(self, inner: Predicate, *, capture_name: str = None, ast_node_generator: ast_node_generator_t = None, syntax_error: str = None):
        super().__init__(capture_name, ast_node_generator, syntax_error)
        self.inner = inner

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        matched, tokens, nodes = self.inner.try_consume(parser, captures)
        return tokens or []

    def __str__(self):
        return f"{self.inner}?" + (f"{{{self.get_capture_name()}}}" if self.get_capture_name() is not None else "")

    def __repr__(self):
        return str(self)


class RepeatingPredicate(Predicate):
    def __init__(self, inner: Predicate, optional: bool = True, *, capture_name: str = None, ast_node_generator: ast_node_generator_t = None, syntax_error: str = None):
        super().__init__(capture_name, ast_node_generator, syntax_error)
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

    def __str__(self):
        return f"{self.inner}{'*' if self.optional else '+'}" + (f"{{{self.get_capture_name()}}}" if self.get_capture_name() is not None else "")

    def __repr__(self):
        return str(self)


class NamedRuleReference(Predicate):
    def __init__(self, rule_name: str, *, capture_name: str = None, ast_node_generator: ast_node_generator_t = None, syntax_error: str = None):
        super().__init__(capture_name, ast_node_generator, syntax_error)
        self.rule_name = rule_name
        self._real_rule = None  # type: Optional[Predicate]

    def try_consume(self, parser: 'SyntaxParser', parent_captures: predicate_captures_t):
        self._update_real_rule(parser)
        return super().try_consume(parser, parent_captures)

    def _update_real_rule(self, parser: 'SyntaxParser'):
        self._real_rule = parser.get_named_rule(self.rule_name)
        if isinstance(self._real_rule, NamedRuleReference):
            self._real_rule._update_real_rule(parser)

    def get_node_generator(self):
        return self._real_rule.get_node_generator()

    def _on_consume_attempt(self, parser: 'SyntaxParser', captures: predicate_captures_t) -> Optional[List[Token]]:
        return self._real_rule._on_consume_attempt(parser, captures)

    def __str__(self):
        return f"{self.rule_name}" + (f"{{{self.get_capture_name()}}}" if self.get_capture_name() is not None else "")

    def __repr__(self):
        return str(self)

opt = OptionalPredicate
seq = SequencePredicate
alt = AlternativePredicate
many = RepeatingPredicate
ref = NamedRuleReference
tk = TerminalPredicate


def real_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        return str({real_str(k): real_str(v) for k, v in obj.items()})

    if isinstance(obj, collections.Iterable):
        return str(list(map(real_str, obj)))

    return str(obj)


class SyntaxParser(object):
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = tokens
        self._pos = 0
        self.rules = {}  # type: Dict[str, Predicate]
        self.root_rule = None

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

    def get_syntax_tree(self):
        captures = {}
        matched, tokens, nodes = self.root_rule.try_consume(self, captures)
        if matched:
            print(f"Syntax parser captures: {real_str(captures)}")

        return matched, tokens, nodes

    def set_root_rule(self, rule_name: str, pred: Predicate):
        self.add_named_rule(rule_name, pred)
        self.root_rule = ref(rule_name, capture_name='root')

    @staticmethod
    def _is_left_recursive(rule_name: str, pred: Predicate):
        if isinstance(pred, SequencePredicate):
            left = pred.preds[0]
            if isinstance(left, NamedRuleReference):
                if left.rule_name == rule_name:
                    return True

        return False

    def _remove_left_recursion(self, rule_name: str, pred: Predicate):
        # remove left recursion
        if isinstance(pred, AlternativePredicate):
            recursive_alternatives = [a for a in pred.alternatives if self._is_left_recursive(rule_name, a)]
            safe_alternatives = [a for a in pred.alternatives if a not in recursive_alternatives]
            if recursive_alternatives and not safe_alternatives:
                raise ValueError(f"Rule {rule_name} has no alternative that is not left recursive")

            if recursive_alternatives:
                print(f"Automatically refactoring rule {rule_name} to remove left recursion.")
                # transform rule of form A ::= A α1 | … | A αm | β1 | … | βn
                # into A ::= β1 A’ | … | βn A’
                # where A’ ::= α1 A’ | … | αm A’ | ε

                prime_rule_name = rule_name + '1'
                prime_alts = []  # α1 A’ ... αm A’
                for old_alt in recursive_alternatives:  # type: SequencePredicate
                    # αi A’
                    prime_alt = seq(*old_alt.preds[1:], ref(prime_rule_name), capture_name=old_alt.get_capture_name(),
                                    ast_node_generator=old_alt.get_node_generator(),
                                    syntax_error=old_alt.get_syntax_error())
                    prime_alts.append(prime_alt)

                prime_rule = opt(alt(*prime_alts))  # A’ ::= α1 A’ | … | αm A’ | ε
                new_alts = [seq(sa, ref(prime_rule_name)) for sa in safe_alternatives]  # β1 A’ .. βn A’
                new_rule = alt(*new_alts, capture_name=pred.get_capture_name(),
                               ast_node_generator=pred.get_node_generator(), syntax_error=pred.get_syntax_error())

                print(f"{pred} was transformed into {new_rule}, where {prime_rule_name} is {prime_rule}")
                pred = new_rule
                self.add_named_rule(prime_rule_name, prime_rule)

        return pred

    def add_named_rule(self, rule_name: str, pred: Predicate):
        pred = self._remove_left_recursion(rule_name, pred)
        self.rules[rule_name] = pred
