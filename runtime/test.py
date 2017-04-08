import builtin
from runtime import stack, machine, instructions

def main():
    #  „v=3;do{put_i(v);v=v-1;}while(v);”
    # Instr * L1;
    # int * v = allocGlobal(sizeof(long
    # int));
    # addInstrA(O_PUSHCT_A, v);
    # addInstrI(O_PUSHCT_I, 3);
    # addInstrI(O_STORE, sizeof(long
    # int));
    # L1 = addInstrA(O_PUSHCT_A, v);
    # addInstrI(O_LOAD, sizeof(long
    # int));
    # addInstrA(O_CALLEXT, requireSymbol( & symbols, "put_i")->addr);
    # addInstrA(O_PUSHCT_A, v);
    # addInstrA(O_PUSHCT_A, v);
    # addInstrI(O_LOAD, sizeof(long
    # int));
    # addInstrI(O_PUSHCT_I, 1);
    # addInstr(O_SUB_I);
    # addInstrI(O_STORE, sizeof(long
    # int));
    # addInstrA(O_PUSHCT_A, v);
    # addInstrI(O_LOAD, sizeof(long
    # int));
    # addInstrA(O_JT_I, L1);
    # addInstr(O_HALT);
    mem = stack.DataStack(8192)
    program = []
    v = mem.alloc(stack.INT.size)

    # v = 3
    program.append(instructions.PUSHCT(v, stack.ADDR, 0))  # push addr of v
    program.append(instructions.PUSHCT(3, stack.INT, 1))  # push constant 3
    program.append(instructions.STORE(stack.INT.size, 2))  # store int to addr

    loop_start = len(program)
    # put_i(v)
    program.append(instructions.PUSHCT(v, stack.ADDR, 3))  # push addr of v
    program.append(instructions.LOAD(stack.INT.size, 4))  # load int from addr
    program.append(instructions.CALLEXT('put_i', 5))  # call put_i

    # v = v - 1
    program.append(instructions.PUSHCT(v, stack.ADDR, 6))  # push addr of v (for store)
    program.append(instructions.PUSHCT(v, stack.ADDR, 6))  # push addr of v (for load)
    program.append(instructions.LOAD(stack.INT.size, 7))  # load int from addr
    program.append(instructions.PUSHCT(1, stack.INT, 8))  # push constant 1
    program.append(instructions.SUB(stack.INT, 9))  # subtract last 2 ints ~ v-1
    program.append(instructions.STORE(stack.INT.size, 10))  # store result back to v

    # while (v != 0) - jump back to loop start
    program.append(instructions.PUSHCT(v, stack.ADDR, 11))  # push addr of v (for load)
    program.append(instructions.LOAD(stack.INT.size, 12))  # load int from addr
    program.append(instructions.PUSHCT(0, stack.INT, 13))  # push constant 0
    program.append(instructions.NOTEQ(stack.INT, 14))  # compare ints
    program.append(instructions.JT(loop_start, 15))  # jump if true
    program.append(instructions.HLT(16))  # end of program

    vm = machine.AtomCVM(mem, program, 0, debug=True)
    vm.execute()
    print(">>>>> PROGRAM HALTED; OUTPUT: \n" + builtin.stdout)


if __name__ == '__main__':
    main()
