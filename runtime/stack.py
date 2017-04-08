import sys
import struct
from typing import Tuple, Union

import runtime.errors, runtime.instructions


class DataStack(object):
    class DataType(object):
        def __init__(self, python_type: type, fmt: str, type_name: str, size: int, valid_range: Tuple[Union[int, float], Union[int, float]]):
            self.python_type = python_type
            self.fmt = fmt
            self.type_name = type_name
            self.size = size
            self.valid_range = valid_range
    
    def __init__(self, size: int):
        self._stack = bytearray(size)
        self._size = size
        self._sp = 0

    def _push_bytes(self, data: bytes):
        if self._sp + len(data) > self._size:
            raise runtime.errors.AtomCVMRuntimeError("out of stack memory")

        self._sp += len(data)
        self._stack[self._sp - len(data):self._sp] = data

    def _pop_bytes(self, count: int) -> bytearray:
        if self._sp - count < 0:
            raise runtime.errors.AtomCVMRuntimeError("stack buffer underflow")

        self._sp -= count
        return self._stack[self._sp:self._sp + count]

    def push(self, value, value_type: DataType):
        if not isinstance(value, value_type.python_type):
            raise runtime.errors.AtomCVMRuntimeError("wrong value type for push")
        try:
            self._push_bytes(struct.pack(value_type.fmt, value))
        except struct.error as e:
            raise runtime.errors.AtomCVMRuntimeError(f"push error: {str(e)}")

    def pop(self, value_type: DataType):
        try:
            return value_type.python_type(struct.unpack(value_type.fmt, bytes(self._pop_bytes(value_type.size)))[0])
        except struct.error as e:
            raise runtime.errors.AtomCVMRuntimeError(f"pop error: {str(e)}")

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

    def alloc(self, size: int):
        if self._sp + size > self._size:
            raise runtime.errors.AtomCVMRuntimeError("out of stack memory")
        ret = self._sp
        self._sp += size
        return ret

    def free(self, size: int):
        if self._sp - size < 0:
            raise runtime.errors.AtomCVMRuntimeError("stack buffer underflow")
        self._sp -= size

    def read_from(self, addr: int, size: int) -> bytes:
        if addr < 0 or size < 0 or addr + size >= self._size:
            raise runtime.errors.AtomCVMRuntimeError("out-of-bounds memory access")
        return bytes(self._stack[addr:addr+size])

    def write_at(self, addr: int, data: bytes):
        if addr < 0 or addr + len(data) >= self._size:
            raise runtime.errors.AtomCVMRuntimeError("out-of-bounds memory access")
        self._stack[addr:addr+len(data)] = data

    def read_string(self, addr: int) -> str:
        if addr < 0:
            raise runtime.errors.AtomCVMRuntimeError("out-of-bounds memory access")
        end = addr
        while self._stack[end] != 0 and end < self._size:
            end += 1
        if end < self._size:
            raise runtime.errors.AtomCVMRuntimeError("stack overflow while reading string")
        return self._stack[addr:end].decode('utf8')

    @property
    def sp(self):
        return self._sp


CHAR = DataStack.DataType.CHAR = DataStack.DataType(int, '=b', 'char', 1, (-2**7, 2**7 - 1))
INT = DataStack.DataType.INT = DataStack.DataType(int, '=i', 'int', 4, (-2**31, 2**31 - 1))
DOUBLE = DataStack.DataType.DOUBLE = DataStack.DataType(float, '=d', 'double', 8, (float('-inf'), float('inf')))
ADDR = DataStack.DataType.ADDR = DataStack.DataType(int, '=i', 'int', 4, (-2**31, 2**31 - 1))


class CallStack(object):
    def __init__(self):
        self.stack = list()

    def call(self, ret_addr: int, frame_pointer: int):
        self.stack.append((ret_addr, frame_pointer))

    @property
    def frame_pointer(self):
        if not self.stack:
            raise runtime.errors.AtomCVMRuntimeError("unbalanced call stack")
        return self.stack[-1][1]

    def ret(self) -> int:
        if not self.stack:
            raise runtime.errors.AtomCVMRuntimeError("unbalanced call stack")
        ret_addr, fp = self.stack.pop()
        return ret_addr
