import sys
import argparse
import random
import tqdm

"""
read queries from file corresponding each line
generate the followng stats:
1. number of lines
2. avg length of line
3. max length of line
4. min length of line
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, required = True)
    args = parser.parse_args()
    n = 0
    avg_len = 0
    max_len = 0
    min_len = 100000000
    with open(args.inp, 'r') as fr:
        for query in tqdm.tqdm(fr):
            query = query.strip()
            n += 1
            avg_len += len(query)
            max_len = max(max_len, len(query))
            min_len = min(min_len, len(query))
    print("Number of lines: {}".format(n))
    print("Avg length of line: {}".format(avg_len / n))
    print("Max length of line: {}".format(max_len))
    print("Min length of line: {}".format(min_len))


if __name__ == '__main__':
    main()