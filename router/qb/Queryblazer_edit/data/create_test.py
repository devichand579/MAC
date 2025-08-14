import sys
import argparse
from numpy import random
import tqdm
import os

"""
read queries from stdin corresponding each line
randomly generate a prefix and print along with query in a single line with a tab
skip queries that do not meet the condition provided
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inp', type=str, default='data/DDC/test.txt')
    parser.add_argument('--out_dir', type=str, default='data/DDC/')
    parser.add_argument('--train_file', type=str, required=True)
    
    args = parser.parse_args()
    random.seed(args.seed)

    args = parser.parse_args()
    
    
    # read line by line and find intersection

    with open(args.inp, 'r') as f:
        file1 = f.readlines()
    with open(args.train_file, 'r') as f:
        file2 = f.readlines()
    file1 = set(file1)
    file2 = set(file2)

    seen = file1.intersection(file2)
    unseen = file1 - seen

    if(not os.path.exists(os.path.join(args.out_dir, 'seen'))):
        os.makedirs(os.path.join(args.out_dir, 'seen'))
    
    
    if(not os.path.exists(os.path.join(args.out_dir, 'unseen'))):
        os.makedirs(os.path.join(args.out_dir, 'unseen'))

    with open(os.path.join(args.out_dir, 'seen/test_formatted.txt'), 'w') as fw:
        fr = seen
        for query in tqdm.tqdm(fr):
            query = query.strip().lower()
            n = len(query)
            if n < 2:
                continue
            for l in range(1, n):
                fw.write('\t'.join([query[:l], query[l:]]) + "\n")
    
    with open(os.path.join(args.out_dir, 'unseen/test_formatted.txt'), 'w') as fw:
        fr = unseen
        for query in tqdm.tqdm(fr):
            query = query.strip().lower()
            n = len(query)
            if n < 2:
                continue
            for l in range(1, n):
                fw.write('\t'.join([query[:l], query[l:]]) + "\n")


if __name__ == '__main__':
    main()