import os

import itertools

import lexer
from syntax import parser, rules


def main():
    for test in os.listdir('tests'):
        print('=================== Analyzing file %s ==================' % test)

        # test = '4.c'
        with open(os.path.join('tests', test), 'rt') as f:
            try:
                tokens = lexer.get_tokens(f)
                for ln, ltokens in itertools.groupby(tokens, lambda t: t.line):
                    print("Line %d: " % ln + ' '.join(map(str, ltokens)))

                syntax_parser = parser.SyntaxParser(tokens)

                for rule_name, pred in rules.syntax_rules.items():
                    syntax_parser.add_named_rule(rule_name, pred)

                syntax_parser.set_root_rule('unit', rules.root_rule)

                matched, tokens, nodes = syntax_parser.get_syntax_tree()
                if matched:
                    print("================== PARSED SYNTAX ================")
                    print(' '.join(map(str, tokens)))
                    print(' '.join(map(str, nodes)))
                else:
                    print("SYNTAX PARSE FAILED")

            except ValueError as e:
                print(e)

        print('=========================================================')


if __name__ == '__main__':
    exit(main())