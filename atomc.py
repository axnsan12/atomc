import os

import itertools
import builtin
import errors
import lexer
import symbols
from syntax import parser, rules


def main():
    for test in os.listdir('tests'):
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

            except errors.AtomCError as e:
                print(e)
                return

        print('=========================================================')


if __name__ == '__main__':
    exit(main())