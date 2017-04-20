import operator
from abc import ABC, abstractmethod
from typing import Union, List

import builtin
import symbols
from runtime import machine, stack, errors

class Instruction(ABC):
    def __init__(self, mnemonic: str, lineno: int):
        self.mnemonic = mnemonic
        self.lineno  = lineno

    @abstractmethod
    def execute(self, vm: 'machine.AtomCVM'):
        raise NotImplementedError("Abstract method.")

    @staticmethod
    def type_suffix(data_type: 'stack.DataStack.DataType'):
        if data_type == stack.CHAR:
            return 'C'
        elif data_type == stack.INT:
            return 'I'
        elif data_type == stack.DOUBLE:
            return 'D'
        elif data_type == stack.ADDR:
            return 'A'
        else:
            raise ValueError("Invalid data type.")

    def __str__(self):
        return self.mnemonic


class ArithmeticInstruction(Instruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', op, mnemonic: str, lineno: int):
        super().__init__(mnemonic, lineno)
        self.data_type = data_type
        self.op = op
        if data_type == stack.ADDR:
            raise errors.AtomCVMRuntimeError("pointer arithmetic is not allowed")

    def _cast_number(self, val: Union[int, float]) -> Union[int, float]:
        if self.data_type == stack.DOUBLE:
            return val

        minv, maxv = self.data_type.valid_range
        if val < minv:
            diff = minv - val
            val = maxv - diff
        elif val > maxv:
            diff = val - maxv
            val = minv + diff

        return val

    def execute(self, vm: 'machine.AtomCVM'):
        try:
            b = vm.data_stack.pop(self.data_type)
            a = vm.data_stack.pop(self.data_type)
            vm.data_stack.push(self._cast_number(self.op(a, b)), self.data_type)
        except ArithmeticError as e:
            raise errors.AtomCVMRuntimeError(str(e))


class ADD(ArithmeticInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__add__, 'ADD_' + self.type_suffix(data_type), lineno)

class SUB(ArithmeticInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__sub__, 'SUB_' + self.type_suffix(data_type), lineno)

class MUL(ArithmeticInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__mul__, 'MUL_' + self.type_suffix(data_type), lineno)

class DIV(ArithmeticInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        op = operator.__truediv__ if data_type == stack.DOUBLE else operator.__floordiv__
        super().__init__(data_type, op, 'DIV_' + self.type_suffix(data_type), lineno)


class LogicalInstruction(Instruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', op, mnemonic: str, lineno: int):
        super().__init__(mnemonic, lineno)
        self.data_type = data_type
        self.op = op

    def execute(self, vm: 'machine.AtomCVM'):
        b = vm.data_stack.pop(self.data_type)
        a = vm.data_stack.pop(self.data_type)
        vm.data_stack.pushi(int(bool(self.op(a, b))))


class AND(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__and__, 'AND_' + self.type_suffix(data_type), lineno)

class OR(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__or__, 'OR_' + self.type_suffix(data_type), lineno)

class EQ(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__eq__, 'EQ_' + self.type_suffix(data_type), lineno)

class NOTEQ(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__ne__, 'NOTEQ_' + self.type_suffix(data_type), lineno)

class GREATER(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__gt__, 'GREATER_' + self.type_suffix(data_type), lineno)

class GREATEREQ(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__ge__, 'GREATEREQ_' + self.type_suffix(data_type), lineno)

class LESS(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__lt__, 'LESS_' + self.type_suffix(data_type), lineno)

class LESSEQ(LogicalInstruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(data_type, operator.__le__, 'LESSEQ_' + self.type_suffix(data_type), lineno)


class NOT(Instruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__('NOT_' + self.type_suffix(data_type), lineno)
        self.data_type = data_type

    def execute(self, vm: 'machine.AtomCVM'):
        val = vm.data_stack.pop(self.data_type)
        vm.data_stack.pushi(int(val == 0))


class NEG(Instruction):
    def __init__(self, data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__('NEG_' + self.type_suffix(data_type), lineno)
        self.data_type = data_type
        if data_type == stack.ADDR:
            raise errors.AtomCVMRuntimeError("pointer arithmetic is not allowed")

    def execute(self, vm: 'machine.AtomCVM'):
        val = vm.data_stack.pop(self.data_type)
        vm.data_stack.push(-val, self.data_type)


class JT(Instruction):
    def __init__(self, addr: int, lineno: int):
        super().__init__('JT', lineno)
        self.addr = addr

    def execute(self, vm: 'machine.AtomCVM'):
        test = vm.data_stack.popi()
        if test != 0:
            vm.ip = self.addr

    def __str__(self):
        return f"{self.mnemonic} {self.addr}"


class JF(Instruction):
    def __init__(self, addr: int, lineno: int):
        super().__init__('JF', lineno)
        self.addr = addr

    def execute(self, vm: 'machine.AtomCVM'):
        test = vm.data_stack.popi()
        if test == 0:
            vm.ip = self.addr

    def __str__(self):
        return f"{self.mnemonic} {self.addr}"


class JMP(Instruction):
    def __init__(self, addr: int, lineno: int):
        super().__init__('JMP', lineno)
        self.addr = addr

    def execute(self, vm: 'machine.AtomCVM'):
        vm.ip = self.addr

    def __str__(self):
        return f"{self.mnemonic} {self.addr}"


class HLT(Instruction):
    def __init__(self, lineno: int):
        super().__init__('HLT', lineno)

    def execute(self, vm: 'machine.AtomCVM'):
        vm.halt()


class NOP(Instruction):
    def __init__(self, lineno: int):
        super().__init__('NOP', lineno)

    def execute(self, vm: 'machine.AtomCVM'):
        pass


class LOAD(Instruction):
    def __init__(self, size: int, lineno: int):
        super().__init__('LOAD', lineno)
        self.size = size

    def execute(self, vm: 'machine.AtomCVM'):
        addr = vm.data_stack.popa()
        data = vm.data_stack.read_from(addr, self.size)
        vm.data_stack.alloc(self.size)
        vm.data_stack.write_at(vm.data_stack.sp - self.size, data)

    def __str__(self):
        return f"{self.mnemonic} {self.size}"


class LEAFP(Instruction):
    def __init__(self, offset: int, lineno: int):
        super().__init__('LEAFP', lineno)
        self.offset = offset

    def execute(self, vm: 'machine.AtomCVM'):
        vm.data_stack.pusha(vm.call_stack.fp + self.offset)

    def __str__(self):
        return f"{self.mnemonic} {self.offset}"


class PUSHCT(Instruction):
    def __init__(self, value: Union[int, float], data_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__('PUSHCT_' + self.type_suffix(data_type), lineno)
        self.value = value
        self.data_type = data_type

    def execute(self, vm: 'machine.AtomCVM'):
        vm.data_stack.push(self.value, self.data_type)

    def __str__(self):
        return f"{self.mnemonic} {self.value}"


class STORE(Instruction):
    def __init__(self, size: int, lineno: int):
        super().__init__('STORE', lineno)
        self.size = size

    def execute(self, vm: 'machine.AtomCVM'):
        data = vm.data_stack.read_from(vm.data_stack.sp - self.size, self.size)
        vm.data_stack.free(self.size)
        addr = vm.data_stack.popa()
        vm.data_stack.write_at(addr, data)

    def __str__(self):
        return f"{self.mnemonic} {self.size}"


class CALL(Instruction):
    def __init__(self, addr: int, lineno: int):
        super().__init__('CALL', lineno)
        self.addr = addr

    def __str__(self):
        return f"{self.mnemonic} {self.addr}"

    def execute(self, vm: 'machine.AtomCVM'):
        vm.call_stack.call(vm.ip + 1)
        vm.ip = self.addr


class ENTER(Instruction):
    def __init__(self, size: int, lineno: int):
        super().__init__('ENTER', lineno)
        self.size = size

    def __str__(self):
        return f"{self.mnemonic} {self.size}"

    def execute(self, vm: 'machine.AtomCVM'):
        vm.data_stack.pusha(vm.call_stack.fp)  # save old frame pointer
        vm.call_stack.fp = vm.data_stack.sp  # set new frame pointer
        vm.data_stack.alloc(self.size)  # alloc space for locals


class DROP(Instruction):
    def __init__(self, size: int, lineno: int):
        super().__init__('DROP', lineno)
        self.size = size

    def execute(self, vm: 'machine.AtomCVM'):
        if self.size > 0:
            vm.data_stack.free(self.size)

    def __str__(self):
        return f"{self.mnemonic} {self.size}"


class OFFSET(Instruction):
    def __init__(self, lineno: int):
        super().__init__('OFFSET', lineno)

    def execute(self, vm: 'machine.AtomCVM'):
        offset = vm.data_stack.popi()
        addr = vm.data_stack.popa()
        vm.data_stack.pusha(addr + offset)


class INSERT(Instruction):
    def __init__(self, where: int, size: int, lineno: int):
        super().__init__('INSERT', lineno)
        self.where = where
        self.size = size

    def execute(self, vm: 'machine.AtomCVM'):
        where = vm.data_stack.sp - self.where
        saved = vm.data_stack.read_from(where, self.where)
        data = saved[-self.size:]
        vm.data_stack.alloc(self.size)
        vm.data_stack.write_at(where, data)
        vm.data_stack.write_at(where + self.size, saved)

    def __str__(self):
        return f"{self.mnemonic} {self.where}, {self.size}"


class RETFP(Instruction):
    def __init__(self, arg_size: int, ret_size: int, lineno: int):
        super().__init__('RETFP', lineno)
        self.arg_size = arg_size
        self.ret_size = ret_size

    def __str__(self):
        return f"{self.mnemonic} {self.arg_size}, {self.ret_size}"

    def execute(self, vm: 'machine.AtomCVM'):
        ret = None
        if self.ret_size > 0:
            ret = vm.data_stack.read_from(vm.data_stack.sp - self.ret_size, self.ret_size)  # get return value from top of stack
        vm.data_stack.free(vm.data_stack.sp - vm.call_stack.fp)  # reset call frame
        vm.call_stack.fp = vm.data_stack.popa()  # restore frame pointer
        if self.arg_size > 0:
            vm.data_stack.free(self.arg_size)  # free function arguments
        if self.ret_size > 0:
            assert ret is not None
            vm.data_stack.alloc(self.ret_size)
            vm.data_stack.write_at(vm.data_stack.sp - self.ret_size, ret)  # put return value on top of stack
        vm.ip = vm.call_stack.ret()  # return control to caller


class CALLEXT(Instruction):
    def __init__(self, builtin_name: str, lineno: int):
        super().__init__('CALLEXT', lineno)
        self.builtin_name = builtin_name

    def __str__(self):
        return f"{self.mnemonic} {self.builtin_name}"

    def execute(self, vm: 'machine.AtomCVM'):
        builtin.exec_builtin(self.builtin_name, vm.data_stack)


class CAST(Instruction):
    def __init__(self, from_type: 'stack.DataStack.DataType', to_type: 'stack.DataStack.DataType', lineno: int):
        super().__init__(f'CAST_{self.type_suffix(from_type)}_{self.type_suffix(to_type)}', lineno)
        self.from_type = from_type
        self.to_type = to_type
        if from_type == stack.ADDR or to_type == stack.ADDR:
            raise errors.AtomCVMRuntimeError(f"CAST cannot operate on addresses")

    def execute(self, vm: 'machine.AtomCVM'):
        val = vm.data_stack.pop(self.from_type)
        vm.data_stack.push(self.to_type.python_type(val), self.to_type)


INT = stack.INT
DOUBLE = stack.DOUBLE
CHAR = stack.CHAR
ADDR = stack.ADDR

data_types = {
    symbols.TYPE_INT: stack.INT,
    symbols.TYPE_REAL: stack.DOUBLE,
    symbols.TYPE_CHAR: stack.CHAR,
}

def add_cast(from_type: symbols.SymbolType, to_type: symbols.SymbolType, lineno: int, program: List[Instruction]):
    assert isinstance(from_type, symbols.PrimitiveType)
    assert isinstance(to_type, symbols.PrimitiveType)
    from_type = data_types[from_type]
    to_type = data_types[to_type]
    if from_type != to_type:
        program.append(CAST(from_type, to_type, lineno))
