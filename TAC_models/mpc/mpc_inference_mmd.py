from tqdm import tqdm
import pickle
import os
import json
import argparse
from utils import QueryCompletion, preprocess, load_text_streamMMD, is_empty, create_suffixes, check_and_create_path, get_line_count
import sys
sys.setrecursionlimit(10000)


# Load the trie from a file
def load_trie_from_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def get_completions(main_trie, suffix_trie, prefix, k_completions=10, suffix_context=2):
    res = []
    main_completions = main_trie.find_completions(prefix)

    main_completions_dict = {}
    for z in main_completions:
        if z[0] in main_completions_dict:
            continue
        main_completions_dict[z[0]]=1
        # adding prefix "MT" to denote main trie completions
        res.append([z[0], "MT:%s" % str(z[1])])

    if suffix_trie is not None and len(res)<k_completions:
        backfill = k_completions - len(res)
        prefix_tokens = prefix.strip().split(' ')
        ends_with_space = " " if prefix[-1]==" " else ""
        suffix_completions = []
        suffix_completions_dict = {}
        
        # minimum words to consider during suffix match
        for idx in range(suffix_context, len(prefix_tokens)):
            suffix = " ".join(prefix_tokens[-idx:]) + ends_with_space
            partial_prefix = " ".join(prefix_tokens[:len(prefix_tokens)-idx])
            for temp_completions in suffix_trie.find_completions(suffix):
                full_completion = partial_prefix + " " + temp_completions[0]
                if full_completion in main_completions_dict or full_completion in suffix_completions_dict:
                    continue
                suffix_completions_dict[full_completion] = 1
                suffix_completions.append((full_completion, temp_completions[1]))
            # print("suffix completions : %d" % len(suffix_completions))
        suffix_completions = sorted(suffix_completions, key=lambda x: x[1], reverse=True)[:backfill]
        if len(suffix_completions)>0:
            for z in suffix_completions:
                # adding prefix "ST" to denote suffix trie completions
                res.append([z[0], "ST:%s" % str(z[1])])
    
    return res

def init(args):
    print("loading main trie from %s" % (args.main_trie))
    query_completion = load_trie_from_file(args.main_trie)

    suffix_completion = None
    if args.suffix_trie is not None:
        print("loading suffix trie from %s" % (args.suffix_trie))
        suffix_completion = load_trie_from_file(args.suffix_trie)

    total_count = get_line_count(args.input_file, args.data_limit)
    inference_count = 0
    with open(args.output_file, 'w', encoding='utf-8') as dfile:
        with tqdm(total=total_count) as pbar:
            for is_valid, input_text, output_text in load_text_streamMMD(args.input_file):
                pbar.update(1)
                if not is_valid:
                    break
                if is_empty(input_text):
                    continue
                prefix = input_text.lstrip().lower().strip('\n')
                # prefix = "sure . but you know that i can't read any c"
                # take last 50 characters within the prefix
                # prefix = prefix[-50:]
                completions = get_completions(query_completion, suffix_completion, prefix, 
                                            args.k_completions, args.suffix_context_words)

                dfile.write(json.dumps({"id":id, "completions": completions}) + "\n")
                inference_count+=1
    print("Total inference: ", inference_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing the prefixes')
    parser.add_argument('--main_trie', type=str, required=True,
                        help='path to location to save the trie object.')
    parser.add_argument('--suffix_trie', type=str, default=None,
                        help='path to location to save the trie object.')
    parser.add_argument("--suffix_context_words", type=int, default=2,
                            help="number of words to consider for suffix completion")
    parser.add_argument('--output_file', type=str, required=True,
                        help='path to location to save the trie object.')
    parser.add_argument('--k_completions', type=int, default=10,
                            help='maximum number of completions to return.')
    parser.add_argument('--data-limit',
                        type=int,
                        default=-1,
                        help='maximum number of instances to load. -1 to include all.')
    args = parser.parse_args()
    check_and_create_path(args.output_file)
    init(args)