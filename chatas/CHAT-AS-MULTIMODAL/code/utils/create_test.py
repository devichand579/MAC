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
    parser.add_argument('--inp', type=str, required = True)
    parser.add_argument('--out_dir', type=str, required = True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--cinp', type=str, default=None)
    
    args = parser.parse_args()

    if(args.context and args.cinp is None):
        print("cinp is required if context is true")
        sys.exit(1)
    
    if(not args.context and args.cinp is not None):
        print("cinp is not required if context is false")
        sys.exit(1)

    random.seed(args.seed)

    
    
    # read line by line and find intersection

    with open(args.inp, 'r') as f:
        file1 = f.readlines()
    with open(args.train_file, 'r') as f:
        file2 = f.readlines()
    if(args.context):
        with open(args.cinp, 'r') as f:
            file3 = f.readlines()
    print("Number of queries in test file: ", len(file1))
    print("Number of queries in train file: ", len(file2))
    if(args.context):
        print("Number of queries in context file: ", len(file3))
        if(len(file1) != len(file3)):
            print("Number of queries in test and context file are not equal")
            sys.exit(1)

    seen = []
    unseen = []

    print("Finding intersection of queries in test and train file")
    for i in (range(len(file1))):
        # print live the length of seen and unseen
        print("Number of seen queries: ", len(seen), ", Number of unseen queries: ", len(unseen), ", percentage done: ", (i/len(file1))*100, "%", end="\r")
        if(args.context):
            mystring = file3[i]
        else:
            mystring = file1[i]
        if(file1[i] in file2):
            seen.append(mystring)
        else:
            unseen.append(mystring)

    print("Number of seen queries: ", len(seen), ", Number of unseen queries: ", len(unseen), ", percentage done: ", (i/len(file1))*100, "%", end="\n")

    if(not os.path.exists(os.path.join(args.out_dir, 'seen'))):
        os.makedirs(os.path.join(args.out_dir, 'seen'))
    if(not os.path.exists(os.path.join(args.out_dir, 'unseen'))):
        os.makedirs(os.path.join(args.out_dir, 'unseen'))

    print("splitting seen and unseen queries into test_formatted.txt")
    ctr = 0
    with open(os.path.join(args.out_dir, 'seen/test_formatted.txt'), 'w') as fw:
        fr = seen
        for query in tqdm.tqdm(fr):
            query = query.strip('\n').lower()
            if(args.context):
                breaking_sentence = query.split("\t")[-1]
                context = query.split("\t")[0]
                n = len(breaking_sentence)
                if n < 2:
                    continue
                for l in range(1, n):
                    ctr += 1
                    fw.write('\t'.join([''.join([context, breaking_sentence[:l]]), breaking_sentence[l:]]) + "\n")
            else:
                n = len(query)
                if n < 2:
                    continue
                for l in range(1, n):
                    ctr += 1
                    fw.write('\t'.join([query[:l], query[l:]]) + "\n")
    print("Number of seen queries: ", ctr)
    ctr = 0
    with open(os.path.join(args.out_dir, 'unseen/test_formatted.txt'), 'w') as fw:
        fr = unseen
        for query in tqdm.tqdm(fr):
            query = query.strip('\n').lower()
            if(args.context):
                breaking_sentence = query.split("\t")[-1]
                context = query.split("\t")[0]
                n = len(breaking_sentence)
                if n < 2:
                    continue
                for l in range(1, n):
                    ctr += 1
                    fw.write('\t'.join([''.join([context, breaking_sentence[:l]]), breaking_sentence[l:]]) + "\n")
            else:
                n = len(query)
                if n < 2:
                    continue
                for l in range(1, n):
                    ctr += 1
                    fw.write('\t'.join([query[:l], query[l:]]) + "\n")
    print("Number of unseen queries: ", ctr)


if __name__ == '__main__':
    main()
