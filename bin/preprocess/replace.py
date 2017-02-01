# -*- coding: utf-8 -*-
"""

"""
import sys
import json
import re

def main(fi):
    num_pat = r'\d'
    num_pat = re.compile(num_pat)

    symbol_pat = r'-'
    symbol_pat = re.compile(symbol_pat)

    for line in fi:
        data = json.loads(line)

        data['question'] = [num_pat.sub(repl='#', string=token) for token in data['question']]
        data['answer'] = [num_pat.sub(repl='#', string=token) for token in data['answer']]
        data['question'] = [symbol_pat.sub(repl='_', string=token) for token in data['question']]
        data['answer'] = [symbol_pat.sub(repl='_', string=token) for token in data['answer']]
        print(json.dumps(data))

if __name__ == "__main__":
    main(sys.stdin)
