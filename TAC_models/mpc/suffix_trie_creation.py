from tqdm import tqdm
import pickle
import os
import argparse
from utils import (
    QueryCompletion,
    preprocess,
    load_text_stream,
    create_suffixes,
    get_line_count,
)
from collections import defaultdict
import sys

sys.setrecursionlimit(10000)

K = 2000


# check whether base_path to a file exists or not and create if it doesn't
def check_and_create_path(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def is_empty(input_text):
    input_text = input_text.strip()
    if input_text == "" or len(input_text) == 0 or len(input_text.split("\t")) <= 1:
        return True
    return False


def init(args):
    query_completion = QueryCompletion()
    suffix_freq = defaultdict(lambda: 0)
    index = 0
    total_count = get_line_count(args.input_file, args.data_limit)
    print("extracting suffixes from %d queries" % (total_count))
    with tqdm(total=total_count) as pbar:
        for is_valid, input_text in load_text_stream(args.input_file, args.data_limit):
            pbar.update(1)
            if not is_valid:
                break
            if is_empty(input_text) or len(input_text) > K:
                continue
            try:
                query, frequency = input_text.split("\t")
            except:
                print("Error in line: ", input_text)
                continue
            suffixes = create_suffixes(preprocess(query))
            for z in suffixes:
                suffix_freq[z] += 1
            query_completion.insert(preprocess(query), int(frequency))
            index += 1
    print("%d suffixes extracted from %d queries" % (len(suffix_freq), index))
    print("creating the suffix trie")
    # creating suffix trie
    ctr = 0
    for suffix, freq in tqdm(suffix_freq.items()):
        if freq < args.suffix_threshold:
            continue
        query_completion.insert(suffix, freq)
        ctr += 1
    print("Total suffixes inserted: ", ctr)
    print("saving the suffix trie")
    with open(args.output_trie, "wb") as f:
        pickle.dump(query_completion, f)
    print("suffix trie saved at: ", args.output_trie)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="file containing the query, frequency pairs.",
    )
    parser.add_argument(
        "--output_trie",
        type=str,
        required=True,
        help="path to location to save the trie object.",
    )
    parser.add_argument(
        "--suffix_threshold",
        type=int,
        default=2,
        help="minimum frequency of suffix to be considered for completion.",
    )
    parser.add_argument(
        "--data-limit",
        type=int,
        default=-1,
        help="maximum number of instances to load. -1 to include all.",
    )
    args = parser.parse_args()
    check_and_create_path(args.output_trie)
    init(args)
