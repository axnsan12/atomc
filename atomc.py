import argparse
import io
import timeit

import acc
import builtin
import errors
import lexer
import symbols
from runtime import stack, machine
from syntax import rules, parser


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('file', help='source code file')
    argp.add_argument('-q', '--quiet', help="Supress program output", action='store_true')
    argp.add_argument('--times', help='execute program TIMES times and print the total execution time', type=int, default=1)
    args = argp.parse_args()
    infile = args.file

    with open(infile, 'rt') as f:
        code = f.read()
        code = io.StringIO(code)

    try:
        tokens = lexer.get_tokens(code)
        syntax_parser = parser.SyntaxParser(tokens)

        for rule_name, pred in rules.syntax_rules.items():
            syntax_parser.add_named_rule(rule_name, pred)
        syntax_parser.set_root_rule('unit', rules.root_rule)

        unit_node = syntax_parser.get_syntax_tree()

        builtin_symbol_table = symbols.SymbolTable('builtin', symbols.StorageType.BUILTIN, None)
        for f in builtin.all_builtins:
            builtin_symbol_table.add_symbol(f)

        global_symbol_table = symbols.SymbolTable('global', symbols.StorageType.GLOBAL, builtin_symbol_table)
        unit_node.bind_symbol_table(global_symbol_table)
        unit_node.validate()

        mem = stack.DataStack(8192)
        program, entry_point = acc.atomc_compile_unit(unit_node, global_symbol_table, mem)

        builtin.stdout = ''
        builtin.interactive = True
        if args.quiet:
            builtin._stdout = lambda *_: None
        vm = machine.AtomCVM(mem, program, entry_point, debug=False)
        scope = globals()
        scope['vm'] = vm
        if args.quiet:
            run_code = "vm.execute(); vm.reset();"
        else:
            run_code = "vm.execute(); print(); vm.reset();"
        etime = timeit.timeit(run_code, globals=scope, number=args.times)
        etime = round(etime, 3)
        if not args.quiet:
            print("-------------------------------------")
        if args.times > 1:
            print(f"Time to execute program {args.times} times: {etime}")
        else:
            print(f"Program execution finished in {etime} seconds")
        return 0
    except errors.AtomCRuntimeError as e:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(e)
        return 1
    except errors.AtomCError as e:
        print(e)
        return 1


if __name__ == '__main__':
    exit(main())
