class AtomCError(Exception):
    pass


class AtomCSyntaxError(AtomCError):
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


class AtomCLexicalError(AtomCError):
    def __init__(self, error_message: str, state: int, line_number: int, partial_token: str):
        self.state = state
        self.line_number = line_number
        self.partial_token = partial_token
        self.error_message = error_message

    def __str__(self):
        return "Lexer error in state %d at line %d (buffer is %r): %s"\
               % (self.state, self.line_number, self.partial_token, self.error_message)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class AtomCTypeError(AtomCError):
    def __init__(self, error_message: str, lineno: int):
        self.error_message = error_message
        self.lineno = lineno

    def __str__(self):
        return f"Type error at line {self.lineno}: {self.error_message}"

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class AtomCDomainError(AtomCError):
    def __init__(self, error_message: str, lineno: int):
        self.error_message = error_message
        self.lineno = lineno

    def __str__(self):
        return f"Domain error at line {self.lineno}: {self.error_message}"

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
