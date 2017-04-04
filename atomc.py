import os

import itertools
import builtin
import lexer
import symbols
from syntax import parser, rules


def main():
    for test in os.listdir('tests'):
        print('=================== Analyzing file %s ==================' % test)

        # test = '4.c'
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
                    print("================== PARSED SYNTAX ================")
                    print(unit_node)
                else:
                    print("SYNTAX PARSE FAILED")

                global_symbol_table = symbols.SymbolTable('global', symbols.StorageType.GLOBAL, None)
                for f in builtin.all_builtins:
                    global_symbol_table.add_symbol(f)

                unit_node.bind_symbol_table(global_symbol_table)
                unit_node.validate()

                print(global_symbol_table)

            except ValueError as e:
                print(e)

        print('=========================================================')


if __name__ == '__main__':
    exit(main())