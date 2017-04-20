from typing import Sequence
from runtime import stack, instructions, errors
import errors as acerrors

class AtomCVM(object):
    def __init__(self, memory: 'stack.DataStack', program: Sequence['instructions.Instruction'], entry_point, debug=False):
        self.globals = memory.copy()
        self.data_stack = memory.copy()
        self.call_stack = stack.CallStack()
        self.program = program
        self.entry_point = entry_point
        self._ip = entry_point
        self._ip_jumped = False
        self._halted = False
        self.debug = debug

        if entry_point not in range(0, len(self.program)):
            raise ValueError("Entry point out of program memory")

    def _print_debug(self, msg: str):
        if self.debug:
            print(msg)

    @property
    def ip(self):
        return self._ip

    @ip.setter
    def ip(self, addr: int):
        if addr not in range(0, len(self.program)):
            raise errors.AtomCVMRuntimeError("Instruction pointer jumped out of program memory")
        self._ip = addr
        self._ip_jumped = True
        self._print_debug(f"ip jumped to {self.ip}")

    def execute(self):
        while not self._halted:
            if self.ip not in range(0, len(self.program)):
                raise errors.AtomCVMRuntimeError("Instruction pointer outside program memory")
            instr = self.program[self._ip]
            self._ip_jumped = False
            self._print_debug(f"executing `{instr}`@{instr.lineno}")
            try:
                instr.execute(self)
            except errors.AtomCVMRuntimeError as e:
                raise acerrors.AtomCRuntimeError(str(e), instr.lineno)
            if not self._ip_jumped:
                self._ip += 1

    def halt(self):
        self._halted = True

    def reset(self):
        self._ip = self.entry_point
        self._halted = self._ip_jumped = False
        self.data_stack = self.globals.copy()
        self.call_stack.reset()
