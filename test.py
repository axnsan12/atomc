import os

import itertools
import builtin
import errors
import lexer
import symbols
from syntax import parser, rules, tree
from runtime import stack, machine, instructions


def main():
    only = ['9.c']
    for test in os.listdir('tests'):
        if only and test not in only:
            continue
        print('================== Analyzing file %s ==================' % test)

        with open(os.path.join('tests', test), 'rt') as infile:
            try:
                tokens = lexer.get_tokens(infile)
                for ln, ltokens in itertools.groupby(tokens, lambda t: t.line):
                    print("Line %d: " % ln + ' '.join(map(str, ltokens)))

                syntax_parser = parser.SyntaxParser(tokens)

                for rule_name, pred in rules.syntax_rules.items():
                    syntax_parser.add_named_rule(rule_name, pred)

                syntax_parser.set_root_rule('unit', rules.root_rule)

                unit_node = syntax_parser.get_syntax_tree()
                if unit_node is not None:
                    print("===================== PARSED SYNTAX =====================")
                    print(unit_node)
                else:
                    print("SYNTAX PARSE FAILED")

                builtin_symbol_table = symbols.SymbolTable('builtin', symbols.StorageType.BUILTIN, None)
                for f in builtin.all_builtins:
                    builtin_symbol_table.add_symbol(f)

                global_symbol_table = symbols.SymbolTable('global', symbols.StorageType.GLOBAL, builtin_symbol_table)
                unit_node.bind_symbol_table(global_symbol_table)
                print(builtin_symbol_table)

                unit_node.validate()
                print("============== TYPE VALIDATION SUCCESSFUL ===============")
                # alloc space for globals
                mem = stack.DataStack(8192)
                globals_size = sum(symbol.type.sizeof for symbol in global_symbol_table.symbols if symbol.storage == symbols.StorageType.GLOBAL)
                global_mem = mem.alloc(globals_size)
                mem.write_at(global_mem, b'\0' * globals_size)

                # write string constants in global memory so they can be passed by address
                for node in unit_node._children:
                    if isinstance(node, tree.ConstantLiteralNode):
                        if isinstance(node.constant_type, symbols.ArrayType):
                            if node.constant_type.elem_type == symbols.TYPE_CHAR:
                                assert node.constant_type.size == len(node.constant_value) + 1
                                node.addr = mem.alloc(node.constant_type.size)
                                mem.write_at(node.addr, node.constant_value.encode('utf8') + b'\0')

                program = []
                unit_node.compile(program)
                main_func = global_symbol_table.get_symbol('main')

                for addr, instr in enumerate(program):
                    print(f"{addr}: {instr} {'<----- ENTRY POINT' if addr == main_func.offset else ''}")
                print("============== COMPILATION SUCCESSFUL ===============")
                print("Running program...")
                builtin.stdout = ''

                entry_point = len(program)
                main_func = global_symbol_table.get_symbol('main')  # type: symbols.FunctionSymbol
                program.append(instructions.CALL(main_func.offset, main_func.lineno))
                program.append(instructions.HLT(-1))
                vm = machine.AtomCVM(mem, program, entry_point, debug=True)
                vm.execute()
                print(">>>>> PROGRAM HALTED; OUTPUT: \n" + builtin.stdout)

            except errors.AtomCError as e:
                print(e)
                return

        print('=========================================================')


if __name__ == '__main__':
    exit(main())