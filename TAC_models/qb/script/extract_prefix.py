'''
/*
* Copyright (c) 2018, salesforce.com, inc.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
* For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
*/
'''

import sys
import argparse
import random
import io


"""
read queries from stdin corresponding each line
randomly generate a prefix and print along with query in a single line with a tab
skip queries that do not meet the condition provided
"""

printable_chars = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'


def is_printable(s):
    return True
    for c in s:
        if c not in printable_chars:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_prefix_len', type=int, default=2)
    parser.add_argument('--min_suffix_len', type=int, default=1)
    parser.add_argument('--n', type=int)
    args = parser.parse_args()
    random.seed(args.seed)

    with io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8') as f:
        for query in f:
            query = query.strip()
            if(not is_printable(query)):
                print('\t'.join(['-']*2))
                continue
            n = len(query)
            if n < args.min_prefix_len + args.min_suffix_len - 1:
                continue
            for kk in range(min(args.n, n - args.min_suffix_len - args.min_prefix_len + 1)):
                l = random.randint(args.min_prefix_len - 1, n - args.min_suffix_len)
                print('\t'.join([query[:l], query]))


if __name__ == '__main__':
    main()