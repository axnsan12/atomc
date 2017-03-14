from plistlib import Dict
from collections import defaultdict
from typing import Optional, Callable, Any, Set, List
from typing import Tuple
from typing import Union
from enum import Enum
import os
import string
import itertools


class TokenType(Enum):
    ID = 1,
    BREAK = 2,
    CHAR = 3,
    DOUBLE = 4,
    ELSE = 5,
    FOR = 6,
    IF = 7,
    INT = 8,
    RETURN = 9,
    STRUCT = 10,
    VOID = 11,
    WHILE = 12,
    CT_INT = 13,
    CT_REAL = 14,
    CT_CHAR = 15,
    CT_STRING = 16,
    COMMA = 17,
    SEMICOLON = 18,
    LPAR = 19,
    RPAR = 20,
    LBRACKET = 21,
    RBRACKET = 22,
    LACC = 23,
    RACC = 24,
    ADD = 25,
    SUB = 26,
    MUL = 27,
    DIV = 28,
    DOT = 29,
    AND = 30,
    OR = 31,
    NOT = 32,
    ASSIGN = 33,
    EQUAL = 34,
    NOTEQ = 35,
    LESS = 36,
    LESSEQ = 37,
    GREATER = 38,
    GREATEREQ = 39,
    END = 40


class Token(object):
    def __init__(self, code: TokenType, value: Optional[Any], line: int):
        assert isinstance(code, TokenType)
        self.code = code  # type: TokenType
        self.value = value
        self.line = line

    def __str__(self):
        s = "%s:%r" % (self.code.name, self.value) if self.value is not None else self.code.name
        return "<" + s + ">"

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class Error(object):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class Tokenizer(object):

    def __init__(self, token_callback: Callable[[Token], None], error_callback: Callable[[int, str, Error], None], debug=False):
        """
        Initialise a new tokenizer.
        The `token_callback` function gets called every time a new token is parsed.
        The `error_callback` function gets called when an error is encountered.
        It receives as parameters the current state and the partial token that was parsed until the error was encountered
        :param token_callback:
        :param error_callback:
        :param debug:
        """
        self.transitions = defaultdict(dict)  # type: Dict[int, Dict[str, Union[int, Error]]]
        self.non_consuming = set()  # type: Set[Tuple[int, int, Optional[str]]]
        self.finalizers = {}  # type: Dict[int, Callable[[int, str], Token]]

        assert token_callback is not None
        assert error_callback is not None
        self.token_callback = token_callback
        self.error_callback = error_callback
        self.debug = debug

        self.current_state = 0  # type: int
        self.buffer = ''  # type: str
        self.error = False
        self.checked = False

    def _print_debug(self, msg: str):
        if self.debug:
            print(msg)

    def _check_valid_transition(self, from_state: int, characters: Optional[str]) -> Optional[str]:
        if from_state in self.finalizers:
            raise ValueError("Cannot add outgoing transitions from final state %d" % from_state)

        if characters is None and None in self.transitions[from_state]:
            raise ValueError("State %d already has a default transition" % from_state)

        if characters:
            for existing in self.transitions[from_state]:
                if existing is not None and existing in characters:
                    raise ValueError("State %d already has a transition defined for character %s" % (from_state, existing))

        return characters

    def reset_state(self):
        self.buffer = ''
        self.current_state = 0
        self.checked = False
        self.error = False

    def add_transition(self, from_state: int, to_state: Union[int, Error], characters: Optional[str], consume):
        """
        Add a transition from `from_state` into `to_state` in the transition table.

        :param from_state: source state number
        :param to_state: target state number
        :param characters: a list characters to transition on; None represents a default transition,
        :param consume: whether to consume the charater that was just read
        :raise ValueError: if any character in `characters` already has a transition defined
        """
        self._check_valid_transition(from_state, characters)
        if isinstance(to_state, Error) and consume:
            consume = False

        if characters is None:
            self.transitions[from_state][characters] = to_state
            if not consume:
                self.non_consuming.add((from_state, to_state, None))
        else:
            for char in characters:
                self.transitions[from_state][char] = to_state
                if not consume:
                    self.non_consuming.add((from_state, to_state, char))


        self.checked = False

    def add_finalizer(self, state: int, finalizer: Callable[[int, str], Token]):
        """
        Define a finalizer for a state to produce a token.
        The finalizer function receives the state number and all characters consumed since leaving state 0.

        :param state: final state
        :param finalizer: finalizer function
        :raise ValueError: if trying to add a finalizer for state 0 or a state that already has a finalizer
        """
        if state in self.finalizers:
            raise ValueError("State %d already has a finalizer")
        if state == 0:
            raise ValueError("Cannot add a finalizer for state 0")
        if len(self.transitions[state]) > 0:
            raise ValueError("Cannot make state %d final because it has outgoing transitions" % state)

        self.add_transition(state, 0, None, consume=False)
        self.finalizers[state] = finalizer

    def _check_reachable_states(self, state: int, reachable: Set[int], dead_end: Set[int]):
        if state not in reachable:
            reachable.add(state)
            if state not in self.transitions:
                dead_end.add(state)
                return

            for next_state in self.transitions[state].values():
                if not isinstance(next_state, Error):
                    self._check_reachable_states(next_state, reachable, dead_end)

    def check(self) -> List[str]:
        """
        Check that the transition table is valid:
            - all existent states have a valid transition for any character (i.e. have a default transition)
            - all existent states are reachable from state 0 throguh at least one transition sequence
        :return: a list of error messages
        """
        errors = []
        all_states = set()
        for state, trans in self.transitions.items():
            all_states.add(state)
            all_states.update(s for s in trans.values() if not isinstance(s, Error))

        for state in all_states:
            trans = self.transitions[state]
            if state not in self.finalizers and state != 0 and None not in trans:
                errors.append("State %d is not final and does not have a default transition. Add an error?" % state)

        reachable = set()
        dead_end = set()
        self._check_reachable_states(0, reachable, dead_end)
        errors.extend("Unreachable state %d" % s for s in all_states - reachable)
        errors.extend("State %d is a dead end" % s for s in dead_end)
        self.checked = True
        return errors

    def _transition_state(self, to_state: Union[int, Error]) -> bool:
        """

        :param to_state:
        :return: transition success state
        """
        if isinstance(to_state, Error):
            self._print_debug("Encountered error %s" % to_state)
            self.error = True
            self.error_callback(self.current_state, self.buffer, to_state)
            return False

        self.current_state = to_state
        if to_state == 0:
            self._print_debug("Discarding buffer %r (transition to 0)" % self.buffer)
            self.buffer = ''

        if to_state in self.finalizers:
            finalizer = self.finalizers[to_state]
            token = finalizer(to_state, self.buffer)
            self._print_debug("Final state {} generated token {} at line {}".format(to_state, token, token.line))
            self.token_callback(token)

        return True

    def feed(self, char: str) -> bool:
        """
        Feed a character into the tokenizer. Transitions from the current state are recursively
            handled until a consuming transition is encountered.
        :param char: character to feed
        :return:
        """
        if not self.checked:
            errors = self.check()
            print('\n'.join(errors))
            assert not errors
        if self.error:
            raise ValueError("Cannot use tokenizer after it entered error state.")

        trans = self.transitions[self.current_state]
        if char in trans:
            next_state = trans[char]
            consuming = (self.current_state, next_state, char) not in self.non_consuming
            self._print_debug("%r: explicit %sconsuming transition from %d to %d"
                              % (char, '' if consuming else 'non-', self.current_state, next_state))
        elif None in trans:
            next_state = trans[None]
            consuming = (self.current_state, next_state, None) not in self.non_consuming
            self._print_debug("%r: default %sconsuming transition from %d to %d"
                              % (char, '' if consuming else 'non-', self.current_state, next_state))
        else:
            raise ValueError("There is no transition defined for %r from state %d" % (char, self.current_state))

        if consuming:
            self.buffer += char

        if self._transition_state(next_state):
             return consuming or self.feed(char)

        return False


def parse_real(real: str) -> float:
    return float(real)


def parse_int(buf: str) -> int:
    if buf.startswith('0x'):
        return int(buf[2:], 16)

    if buf.startswith('0') and len(buf) > 1:
        return int(buf[1:], 8)

    return int(buf)


def unescape(s: str) -> str:
    return s.encode('utf8').decode('unicode_escape')


def get_tokens(fobj: 'file object') -> List[Token]:
    tokens = []
    tokenizer = Tokenizer(lambda tok: tokens.append(tok),
                          lambda state, partial, err: print("Error in state %d at line %d (buffer is %r): %s"
                                                            % (state, line_number, partial, err)),
                          debug=False)

    keywords = {'break', 'char', 'double', 'else', 'for', 'if', 'int', 'return', 'struct', 'void', 'while'}

    # ID
    tokenizer.add_transition(0, 20, string.ascii_letters + '_', consume=True)
    tokenizer.add_transition(20, 20, string.ascii_letters + string.digits + '_', consume=True)
    tokenizer.add_transition(20, 21, None, consume=False)

    tokenizer.add_finalizer(21, lambda state, buffer: Token(
        TokenType[buffer.upper()] if buffer.lower() in keywords else TokenType.ID,
        buffer if buffer.lower() not in keywords else None, line_number))

    # SPACE
    tokenizer.add_transition(0, 0, ' \t\r\n', consume=True)

    # CT_CHAR
    tokenizer.add_transition(0, 14, '\'', consume=True)
    tokenizer.add_transition(14, Error("Character literal must not be empty"), '\'', consume=False)
    tokenizer.add_transition(14, 16, '\\', consume=True)
    tokenizer.add_transition(14, 17, None, consume=True)
    tokenizer.add_transition(16, Error("Invalid escape sequence"), None, consume=False)
    tokenizer.add_transition(16, 17, "abfnrtv'?\"\\0", consume=True)
    tokenizer.add_transition(17, 51, '\'', consume=True)
    tokenizer.add_transition(17, Error("Character literal with multiple elements"), None, consume=False)

    tokenizer.add_finalizer(51, lambda state, buffer: Token(TokenType.CT_CHAR, unescape(buffer[1:-1]), line_number))

    # CT_STRING
    tokenizer.add_transition(0, 15, '"', consume=True)
    tokenizer.add_transition(15, 18, '\\', consume=True)
    tokenizer.add_transition(15, 19, '"', consume=True)
    tokenizer.add_transition(15, 15, None, consume=True)
    tokenizer.add_transition(18, 15, "abfnrtv'?\"\\0", consume=True)
    tokenizer.add_transition(18, Error("Invalid escape sequence"), None, consume=False)

    tokenizer.add_finalizer(19, lambda state, buffer: Token(TokenType.CT_STRING, unescape(buffer[1:-1]), line_number))


    # CT_INT
    tokenizer.add_transition(0, 1, '0', consume=True)
    tokenizer.add_transition(1, 2, string.octdigits, consume=True)
    tokenizer.add_transition(2, 2, string.octdigits, consume=True)

    tokenizer.add_transition(1, 3, 'x', consume=True)
    tokenizer.add_transition(3, 4, string.hexdigits, consume=True)
    tokenizer.add_transition(4, 4, string.hexdigits, consume=True)
    tokenizer.add_transition(3, Error("Invalid hex literal. Hexadecimal digit expected."), None, consume=False)

    tokenizer.add_transition(0, 5, string.digits[1:], consume=True)
    tokenizer.add_transition(5, 5, string.digits, consume=True)

    tokenizer.add_transition(1, 6, None, consume=False)
    tokenizer.add_transition(2, 6, None, consume=False)
    tokenizer.add_transition(4, 6, None, consume=False)
    tokenizer.add_transition(5, 6, None, consume=False)

    tokenizer.add_finalizer(6, lambda state, buffer: Token(TokenType.CT_INT, parse_int(buffer), line_number))

    # CT_REAL
    tokenizer.add_transition(1, 7, '89', consume=True)
    tokenizer.add_transition(2, 7, '89', consume=True)
    tokenizer.add_transition(7, 7, string.digits, consume=True)
    tokenizer.add_transition(7, Error("Invalid octal literal."), None, consume=False)

    tokenizer.add_transition(1, 8, '.', consume=True)
    tokenizer.add_transition(5, 8, '.', consume=True)
    tokenizer.add_transition(7, 8, '.', consume=True)
    tokenizer.add_transition(8, Error("Invalid float/double literal. Digit expected."), None, consume=False)

    tokenizer.add_transition(5, 10, 'eE', consume=True)
    tokenizer.add_transition(7, 10, 'eE', consume=True)
    tokenizer.add_transition(9, 10, 'eE', consume=True)
    tokenizer.add_transition(10, Error("Invalid float/double literal. Digit expected."), None, consume=False)

    tokenizer.add_transition(8, 9, string.digits, consume=True)
    tokenizer.add_transition(9, 9, string.digits, consume=True)

    tokenizer.add_transition(10, 11, '+-', consume=True)
    tokenizer.add_transition(11, Error("Invalid float/double literal. Digit expected."), None, consume=False)

    tokenizer.add_transition(10, 12, string.digits, consume=True)
    tokenizer.add_transition(11, 12, string.digits, consume=True)
    tokenizer.add_transition(12, 12, string.digits, consume=True)

    tokenizer.add_transition(9, 13, None, consume=False)
    tokenizer.add_transition(12, 13, None, consume=False)

    tokenizer.add_finalizer(13, lambda state, buffer: Token(TokenType.CT_REAL, parse_real(buffer), line_number))

    # SINGLE CHARACTERS
    singles = [
        (22, ',', "COMMA"),
        (23, ';', "SEMICOLON"),
        (25, '(', "LPAR"),
        (24, ')', "RPAR"),
        (26, '[', "LBRACKET"),
        (27, ']', "RBRACKET"),
        (28, '{', "LACC"),
        (29, '}', "RACC"),
        (30, '+', "ADD"),
        (31, '-', "SUB"),
        (32, '*', "MUL"),
        (34, '.', "DOT"),
    ]

    for sid, char, code in singles:
        tokenizer.add_transition(0, sid, char, consume=True)
        tokenizer.add_finalizer(sid, lambda state, buffer, _code=code: Token(TokenType[_code], None, line_number))

    # DOUBLE REPEATS
    doubles = [
        (35, 36, '&', 'AND'),
        (37, 38, '|', 'OR')
    ]

    for s1, s2, char, code in doubles:
        tokenizer.add_transition(0, s1, char, consume=True)
        tokenizer.add_transition(s1, s2, char, consume=True)
        tokenizer.add_transition(s1, Error("Single %r encountered. Expected second %r" % (char, char)), None, False)
        tokenizer.add_finalizer(s2, lambda state, buffer, _code=code: Token(TokenType[_code], None, line_number))

    # DOUBLE CHARACTER FORKS
    forks = [
        (39, '!', 40, 'NOT', '=', 41, 'NOTEQ'),
        (42, '=', 53, 'ASSIGN', '=', 43, 'EQUAL'),
        (44, '<', 56, 'LESS', '=', 45, 'LESSEQ'),
        (46, '>', 47, 'GREATER', '=', 48, 'GREATEREQ'),
    ]

    for intermediary, char1, s1, code1, char2, s2, code2 in forks:
        tokenizer.add_transition(0, intermediary, char1, consume=True)
        tokenizer.add_transition(intermediary, s1, None, consume=False)
        tokenizer.add_transition(intermediary, s2, char2, consume=True)

        tokenizer.add_finalizer(s1, lambda state, buffer, _code=code1: Token(TokenType[_code], None, line_number))
        tokenizer.add_finalizer(s2, lambda state, buffer, _code=code2: Token(TokenType[_code], None, line_number))

    # COMMENTS
    tokenizer.add_transition(0, 33, '/', consume=True)

    # line comment
    tokenizer.add_transition(33, 52, '/', consume=True)  # //
    tokenizer.add_transition(52, 0, '\r\n\0', consume=False)
    tokenizer.add_transition(52, 52, None, consume=True)

    # multiline comment
    tokenizer.add_transition(33, 54, '*', consume=True)  # /*
    tokenizer.add_transition(54, 55, '*', consume=True)
    tokenizer.add_transition(54, 54, None, consume=True)
    tokenizer.add_transition(55, 0, '/', consume=True)  # */
    tokenizer.add_transition(55, 55, '*', consume=True)
    tokenizer.add_transition(55, 54, None, consume=True)

    # actual division
    tokenizer.add_transition(33, 50, None, consume=False)
    tokenizer.add_finalizer(50, lambda state, buffer: Token(TokenType.DIV, None, line_number))

    tokenizer.add_transition(0, -1, '\0', consume=True)
    tokenizer.add_finalizer(-1, lambda state, buffer: Token(TokenType.END, None, line_number))

    try:
        for line_number, line in enumerate(fobj):
            for column, char in enumerate(line):
                tokenizer._print_debug("# FEED %r" % char)
                tokenizer.feed(char)

        tokenizer._print_debug("# FEED %r" % '\0')
        tokenizer.feed('\0')
    except ValueError:
        raise

    return tokens


def main():
    for test in os.listdir('tests'):
        print('=================== Analyzing file %s ==================' % test)

        with open(os.path.join('tests', test), 'rt') as f:
            try:
                tokens = get_tokens(f)
                for ln, ltokens in itertools.groupby(tokens, lambda t: t.line):
                    print("Line %d: " % ln + ' '.join(map(str, ltokens)))
            except ValueError as e:
                print(e)

        print('=========================================================')

if __name__ == '__main__':
    exit(main())
