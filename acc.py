from typing import List, Tuple

import errors
import symbols
from runtime import stack, instructions
from syntax import tree


def atomc_compile_unit(syntax_tree: tree.UnitNode, global_symbol_table: symbols.SymbolTable, memory: stack.DataStack) -> Tuple[List[instructions.Instruction], int]:
    # alloc space for globals
    globals_size = sum(symbol.type.sizeof for symbol in global_symbol_table.symbols if
                       symbol.storage == symbols.StorageType.GLOBAL)
    global_mem = memory.alloc(globals_size)
    memory.write_at(global_mem, b'\0' * globals_size)

    # write string constants in global memory so they can be passed by address
    for node in syntax_tree._children:
        if isinstance(node, tree.ConstantLiteralNode):
            if isinstance(node.constant_type, symbols.ArrayType):
                if node.constant_type.elem_type == symbols.TYPE_CHAR:
                    assert node.constant_type.size == len(node.constant_value) + 1
                    node.addr = memory.alloc(node.constant_type.size)
                    memory.write_at(node.addr, node.constant_value.encode('utf8') + b'\0')

    program = []
    syntax_tree.compile(program)

    entry_point = len(program)
    main_func = global_symbol_table.get_symbol('main')  # type: symbols.FunctionSymbol
    program.append(instructions.CALL(main_func.offset, main_func.lineno))
    program.append(instructions.HLT(-1))
    return program, entry_point