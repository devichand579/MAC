import argparse
from copy import deepcopy
import os
import sys
import json
from tqdm import tqdm
import pickle
from utils import QueryCompletion, preprocess, load_text_stream, is_empty, create_suffixes, check_and_create_path, get_line_count
import re
import math
from datetime import datetime
import multiprocessing as mp
sys.setrecursionlimit(1000)


# Load the trie from a file
def load_trie_from_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# modifiy the load_text_stream to function to have begin and end indexes
def load_text_stream_with_index(load_path, max_limit=-1, start_index:int=0):
    index = -1
    with open(load_path, 'r', encoding='utf-8') as f:
        for row in f:
            index+=1
            if index<start_index:
                continue
            input_text = row
            if input_text[-1]=='\n':
                input_text = input_text[:-1]
            if (max_limit>0 and index>=max_limit):
                yield False, "", index
            yield True, input_text, index
    yield False, "", index

def get_completions(main_trie, suffix_trie, prefix, completions_file, k_completions=10, suffix_context=2):
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
    with open(completions_file, 'a') as f:
        f.write(f'{prefix}\t{res}\n')
    return res

# def get_completions(main_trie, suffix_trie, prefix, k_completions=10, suffix_context=2):
#     res = []
#     main_completions = main_trie.find_completions(prefix)

#     main_completions_dict = {}
#     for z in main_completions:
#         if z[0] in main_completions_dict:
#             continue
#         main_completions_dict[z[0]]=1
#         # adding prefix "MT" to denote main trie completions
#         res.append([z[0], "MT:%s" % str(z[1])])

#     if suffix_trie is not None and len(res)<k_completions:
#         backfill = k_completions - len(res)
#         prefix_tokens = prefix.strip().split(' ')
#         ends_with_space = " " if prefix[-1]==" " else ""
#         suffix_completions = []
#         suffix_completions_dict = {}
        
#         # minimum words to consider during suffix match
#         for idx in range(suffix_context, len(prefix_tokens)):
#             suffix = " ".join(prefix_tokens[-idx:]) + ends_with_space
#             partial_prefix = " ".join(prefix_tokens[:len(prefix_tokens)-idx])
#             for temp_completions in suffix_trie.find_completions(suffix):
#                 full_completion = partial_prefix + " " + temp_completions[0]
#                 if full_completion in main_completions_dict or full_completion in suffix_completions_dict:
#                     continue
#                 suffix_completions_dict[full_completion] = 1
#                 suffix_completions.append((full_completion, temp_completions[1]))
#             # print("suffix completions : %d" % len(suffix_completions))
#         suffix_completions = sorted(suffix_completions, key=lambda x: x[1], reverse=True)[:backfill]
#         if len(suffix_completions)>0:
#             for z in suffix_completions:
#                 # adding prefix "ST" to denote suffix trie completions
#                 res.append([z[0], "ST:%s" % str(z[1])])
#     return res

def store_jsonl(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as dfile:
        for line in res:
            json.dump(line, dfile, ensure_ascii=False)
            dfile.write('\n')
    print("written %d lines to json file : %s" % (len(res), file_path))      

def inference(args):
    print("[worker-%d] loading main trie from %s" % (args.worker_id, args.main_trie))
    query_completion = load_trie_from_file(args.main_trie)

    suffix_completion = None
    if args.suffix_trie is not None:
        print("[worker-%d] loading suffix trie from %s" % (args.worker_id, args.suffix_trie))
        suffix_completion = load_trie_from_file(args.suffix_trie)

    output_file = open(args.output_file, 'w', encoding='utf-8')
    prefix_count = args.prefix_count
    print("[worker-%d] prefixes for inference : %d" % (args.worker_id, prefix_count))
    print("[worker-%d] prefixes start index : %d, end_index : %d" % (args.worker_id, args.start_index, args.max_limit))
    args.inference_count = 0
    args.trimmed_prefixes = 0

    stream = load_text_stream_with_index(args.input_file, args.max_limit, args.start_index)
    with tqdm(total=prefix_count, desc="inference", unit="lines", position=0, leave=True) as pbar:
        for status, prefix_data, _ in stream: 
            pbar.update(1)
            if status==False:
                break
            if is_empty(prefix_data):
                continue
            prefix = prefix_data.lstrip().lower().strip('\n')
            # take last 50 characters within the prefix
            # prefix = prefix[-50:]
            completions = get_completions(query_completion, suffix_completion, prefix, 
                                        args.k_completions, args.suffix_context_words)
            output_file.write(json.dumps({"prefix": prefix, "completions": completions}) + "\n")
            args.inference_count+=1
    output_file.close()
    print("[worker-%d] number of inference_count : %d" % (args.worker_id, args.inference_count))

# function to delete the file if it exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# function to get the filename and parent directory given the complete file path
def create_worker_file_name(idx, file_path):
    file_name = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"worker-{idx}-{file_name}")

# function to print time delta in human readable format hh:mm:ss
def get_time_delta(time_delta):
    hours, rem = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global model configuration
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
    parser.add_argument('--completions_file',
                        type=str,
                        default=None,
                        help='path to location to save the trie object.')
    parser.add_argument('--context', default=0, type=int)
    # if multiple cpu cores are available, set num_workers to number of CPU cores available
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='total number of workers')                                     
    args = parser.parse_args()

    global_start_time = datetime.utcnow()
    print("[global] argument passed :")
    for arg in vars(args):
        print("[global] %s - %s" % (arg, getattr(args, arg)))
    print('--'*30)
    args.global_prefix_count = get_line_count(args.input_file, args.data_limit)
    print("[global] total prefix count : %d" % args.global_prefix_count)
    inference_count = 0
    trimmed_prefix_count = 0
    if args.num_workers==1:
        args.worker_id = 0
        args.start_index = 0
        args.prefix_count = args.global_prefix_count
        args.max_limit = args.data_limit
        inference(args)
    else:
        bucket_size = math.ceil(args.global_prefix_count/args.num_workers)
        worker_configs = []
        # creating the config
        for idx in range(args.num_workers):
            worker_config = deepcopy(args)
            worker_config.worker_id = idx
            worker_config.start_index = idx*bucket_size
            worker_config.max_limit = min(worker_config.start_index + bucket_size, args.global_prefix_count)
            worker_config.prefix_count = (worker_config.data_limit - worker_config.start_index)
            worker_config.output_file = create_worker_file_name(idx, os.path.abspath(args.output_file))
            worker_configs.append(worker_config)
        # spawning the process
        worker_processes = []
        print("[global] spawning %d inference processes" % args.num_workers)
        for idx in range(args.num_workers):
            p = mp.Process(target=inference, args=(worker_configs[idx],))
            p.start()
            worker_processes.append(p)
        # wait for all the process to finish
        for p in worker_processes:
            p.join()
        
        # merge intermediate outputs to final file 
        final_output_file = open(args.output_file, 'w', encoding='utf-8')
        for wc in worker_configs:
            stream = load_text_stream(wc.output_file)
            for status, data in stream:
                if status==False:
                    break
                final_output_file.write("%s\n" % data)
        final_output_file.close()
        
        # delete the intermediate files 
        for wc in worker_configs:
            delete_file_if_exists(wc.output_file)

    time_delta = (datetime.utcnow() - global_start_time)
    print("[global] completed %d inference in %s" % (args.global_prefix_count, get_time_delta(time_delta)))
