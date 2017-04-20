import struct
import itertools
from typing import Tuple, Union, Callable
from runtime import errors


def _sentinel(val: int, length: int, val_size: int=4):
    values = tuple(val.to_bytes(val_size, byteorder='big', signed=False))
    yield from itertools.chain(*itertools.repeat(values, length // len(values)))
    yield from values[:length % len(values)]


def print_debug(func: Callable):
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        st = args[0]  # type: DataStack

        if st.debug:
            args_print = ' '.join(map(str, args[1:]))
            ret = f'-> {result}' if result is not None else ''
            print(func.__name__, args_print, ret)

        return result

    return wrapped


class DataStack(object):
    class DataType(object):
        def __init__(self, python_type: type, fmt: str, type_name: str, size: int, valid_range: Tuple[Union[int, float], Union[int, float]]):
            self.python_type = python_type
            self.fmt = fmt
            self.type_name = type_name
            self.size = size
            self.valid_range = valid_range

        def __str__(self):
            return self.type_name
    
    def __init__(self, size: int, debug=False):
        self._stack = bytearray(size)
        self._size = size
        self._sp = 0
        self.debug = debug

    def _push_bytes(self, data: bytes):
        if self._sp + len(data) > self._size:
            raise errors.AtomCVMRuntimeError("out of stack memory")

        self._sp += len(data)
        self._stack[self._sp - len(data):self._sp] = data

    def _pop_bytes(self, count: int) -> bytearray:
        if self._sp - count < 0:
            raise errors.AtomCVMRuntimeError("stack buffer underflow")

        self._sp -= count
        return self._stack[self._sp:self._sp + count]

    @print_debug
    def push(self, value, value_type: DataType):
        if not isinstance(value, value_type.python_type):
            raise errors.AtomCVMRuntimeError("wrong value type for push")
        try:
            self._push_bytes(struct.pack(value_type.fmt, value))
        except struct.error as e:
            raise errors.AtomCVMRuntimeError(f"push error: {str(e)}")

    @print_debug
    def pop(self, value_type: DataType):
        try:
            return value_type.python_type(struct.unpack(value_type.fmt, bytes(self._pop_bytes(value_type.size)))[0])
        except struct.error as e:
            raise errors.AtomCVMRuntimeError(f"pop error: {str(e)}")

    def pushi(self, val: int):
        self.push(val, DataStack.DataType.INT)

    def popi(self) -> int:
        return self.pop(DataStack.DataType.INT)

    def pushd(self, val: float):
        self.push(val, DataStack.DataType.DOUBLE)

    def popd(self) -> float:
        return self.pop(DataStack.DataType.DOUBLE)

    def pusha(self, val: int):
        self.push(val, DataStack.DataType.ADDR)

    def popa(self) -> int:
        return self.pop(DataStack.DataType.ADDR)

    def pushc(self, val: int):
        self.push(val, DataStack.DataType.CHAR)

    def popc(self) -> int:
        return self.pop(DataStack.DataType.CHAR)

    @print_debug
    def alloc(self, size: int):
        if size < 0:
            raise errors.AtomCVMRuntimeError("negative size")
        if self._sp + size > self._size:
            raise errors.AtomCVMRuntimeError("out of stack memory")
        ret = self._sp
        self._stack[self._sp:self._sp+size] = _sentinel(0xBAADF00D, size)
        self._sp += size
        return ret

    @print_debug
    def free(self, size: int):
        if size < 0:
            raise errors.AtomCVMRuntimeError("negative size")
        if self._sp - size < 0:
            raise errors.AtomCVMRuntimeError("stack buffer underflow")
        self._sp -= size
        self._stack[self._sp:self._sp+size] = _sentinel(0xDEADBEEF, size)

    @print_debug
    def read_from(self, addr: int, size: int) -> bytes:
        if size < 0:
            raise errors.AtomCVMRuntimeError("negative size")
        if addr < 0 or size < 0 or addr + size >= self._size:
            raise errors.AtomCVMRuntimeError("out-of-bounds memory access")
        return bytes(self._stack[addr:addr+size])

    @print_debug
    def write_at(self, addr: int, data: bytes):
        if addr < 0 or addr + len(data) >= self._size:
            raise errors.AtomCVMRuntimeError("out-of-bounds memory access")
        self._stack[addr:addr+len(data)] = data

    @print_debug
    def read_string(self, addr: int) -> str:
        if addr < 0:
            raise errors.AtomCVMRuntimeError("out-of-bounds memory access")
        end = addr
        while self._stack[end] != 0 and end < self._size:
            end += 1
        if end >= self._size:
            raise errors.AtomCVMRuntimeError("stack overflow while reading string")
        return self._stack[addr:end].decode('utf8')

    @property
    def sp(self):
        return self._sp

    def copy(self):
        new_stack = DataStack(self._size)
        new_stack._stack[:] = self._stack[:]
        new_stack._sp = self._sp
        return new_stack


CHAR = DataStack.DataType.CHAR = DataStack.DataType(int, '=B', 'char', 1, (-2**7, 2**7 - 1))
INT = DataStack.DataType.INT = DataStack.DataType(int, '=i', 'int', 4, (-2**31, 2**31 - 1))
DOUBLE = DataStack.DataType.DOUBLE = DataStack.DataType(float, '=d', 'double', 8, (float('-inf'), float('inf')))
ADDR = DataStack.DataType.ADDR = DataStack.DataType(int, '=i', 'int', 4, (-2**31, 2**31 - 1))


class CallStack(object):
    def __init__(self):
        self.stack = []
        self._fp = -1

    def call(self, ret_addr: int):
        self.stack.append(ret_addr)

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, frame_pointer: int):
        self._fp = frame_pointer

    def ret(self) -> int:
        if not self.stack:
            raise errors.AtomCVMRuntimeError("unbalanced call stack")
        return self.stack.pop()

    def reset(self):
        self.stack = []
