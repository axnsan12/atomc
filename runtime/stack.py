import sys
import struct
import runtime.errors


class DataStack(object):
    class DataType(object):
        def __init__(self, python_type: type, fmt: str, type_name: str, size: int):
            self.python_type = python_type
            self.fmt = fmt
            self.type_name = type_name
            self.size = size
    
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

    def _push(self, value, value_type: DataType):
        if not isinstance(value, value_type.python_type):
            raise runtime.errors.AtomCVMRuntimeError("wrong value type for push")
        try:
            self._push_bytes(struct.pack(value_type.fmt, value))
        except struct.error as e:
            raise runtime.errors.AtomCVMRuntimeError(f"push error: {str(e)}")

    def _pop(self, value_type: DataType):
        try:
            return value_type.python_type(struct.unpack(value_type.fmt, bytes(self._pop_bytes(value_type.size))))
        except struct.error as e:
            raise runtime.errors.AtomCVMRuntimeError(f"pop error: {str(e)}")

    def pushi(self, val: int):
        self._push(val, DataStack.DataType.INT)

    def popi(self) -> int:
        return self._pop(DataStack.DataType.INT)

    def pushd(self, val: float):
        self._push(val, DataStack.DataType.DOUBLE)

    def popd(self) -> float:
        return self._pop(DataStack.DataType.DOUBLE)

    def pusha(self, val: int):
        self._push(val, DataStack.DataType.DOUBLE)

    def popa(self) -> int:
        return self._pop(DataStack.DataType.DOUBLE)

    def pushc(self, val: int):
        self._push(val, DataStack.DataType.CHAR)

    def popc(self) -> int:
        return self._pop(DataStack.DataType.CHAR)

DataStack.DataType.CHAR = DataStack.DataType(int, '=b', 'char', 1)
DataStack.DataType.INT = DataStack.DataType(int, '=i', 'int', 4)
DataStack.DataType.DOUBLE = DataStack.DataType(float, '=d', 'double', 8)
DataStack.DataType.ADDR = DataStack.DataType.INT