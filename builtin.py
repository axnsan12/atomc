# void put_s(char s[]) Afiseaza sirul de caractere dat
# void get_s(char s[]) Cere de la tastatura un sir de caractere si il depune in s
# void put_i(int i) Afiseaza intregul „i”
# int get_i() Cere de la tastatura un intreg
# void put_d(double d) Afiseaza numarul real „d”
# double get_d() Cere de la tastatura un numar real
# void put_c(char c) Afiseaza caracterul „i”
# char get_c() Cere de la tastatura un caracter
# double seconds() Returneaza un numar (posibil zecimal pentru precizie mai buna)
# de secunde. Nu se specifica de cand se calculeaza acest numar
# (poate fi de la inceputul rularii programului, de la 1/1/1970, …)
import symbols

def _array(elem_type: symbols.BasicType, size: int=None):
    return symbols.ArrayType(elem_type, size)

def _arg_array(name: str, elem_type: symbols.BasicType, size: int=None):
    return symbols.VariableSymbol(name, _array(elem_type, size), symbols.StorageType.ARG)

def _arg_basic(name: str, arg_type: symbols.BasicType):
    return symbols.VariableSymbol(name, arg_type, symbols.StorageType.ARG)

def _builtin(name: str, return_type: symbols.SymbolType, *args: symbols.VariableSymbol):
    return symbols.FunctionSymbol(name, return_type, symbols.StorageType.BUILTIN, *args)


put_s = _builtin('put_s', symbols.TYPE_VOID, _arg_array('s', symbols.TYPE_CHAR))
get_s = _builtin('get_s', _array(symbols.TYPE_CHAR), _arg_array('s', symbols.TYPE_CHAR))

put_i = _builtin('put_i', symbols.TYPE_VOID, _arg_basic('i', symbols.TYPE_INT))
get_i = _builtin('get_i', symbols.TYPE_INT)

put_d = _builtin('put_d', symbols.TYPE_VOID, _arg_basic('d', symbols.TYPE_REAL))
get_d = _builtin('get_d', symbols.TYPE_REAL)

put_c = _builtin('put_c', symbols.TYPE_VOID, _arg_basic('i', symbols.TYPE_CHAR))
get_c = _builtin('get_c', symbols.TYPE_CHAR)

seconds = _builtin('seconds', symbols.TYPE_REAL)

all_builtins = [put_s, get_s, put_i, get_i, put_d, get_d, put_c, get_c, seconds]