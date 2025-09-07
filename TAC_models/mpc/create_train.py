import sys

import utils
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, default='data/DDC/train.txt')
    parser.add_argument('--out', type=str, default='train.mpc')
    args = parser.parse_args()
    mp = {}
    with open(args.inp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = utils.preprocess(line)
            if line in mp:
                mp[line] += 1
            else:
                mp[line] = 1
    with open(args.out, 'w') as f:
        for i in mp:
            f.write('\t'.join([i, str(mp[i])]) + '\n')


if __name__ == '__main__':
    main()