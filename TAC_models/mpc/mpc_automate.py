# automate things
import os
import time
import datetime
import argparse


def output_map(inp):
    if('unseen' in inp and 'DDC' in inp):
        return "unseen.ddc"
    if('unseen' in inp and 'DSTC7' in inp):
        return "unseen.dstc7"
    
    # seen is already there in unseen as un`seen`
    if('seen' in inp and 'DDC' in inp):
        return "seen.ddc"
    if('seen' in inp and 'DSTC7' in inp):
        return "seen.dstc7"
    return "unknown"




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args = args.parse_args()
    # print(('dir {}'.format(os.path.join(args.data, 'seen/test_formatted.txt'))))
    # exit(0)
    cases = [('data/DDC', True), ('data/DDC', False), ('data/DSTC7', True), ('data/DSTC7', False)]
    for case in cases:
        args.data = case[0]
        args.suffix = case[1]
        if args.suffix:
            # run commands
            os.system('python code/mpc/create_train.py --inp {} --out ./train.mpc'.format(os.path.join(args.data, 'train.txt')))
            os.system('python code/mpc/main_trie_creation.py --input_file ./train.mpc --output_trie ./main.mpc --threshold 0')
            os.system('python code/mpc/suffix_trie_creation.py --input_file ./train.mpc --output_trie ./suffix.mpc --suffix_threshold 2')
            os.system('python code/mpc/mpc_inference_orig.py --input_file {} --main_trie ./main.mpc --suffix_trie ./suffix.mpc --output_file ./completions.mpc'.format(os.path.join(args.data, 'seen/test_formatted.txt')))
            os.system('python code/mpc/transform2eval.py --input_file ./completions.mpc --output_file {}'.format('out.' + output_map(os.path.join(args.data, 'seen/test_formatted.txt')) + '.mpc.suffix'))
            os.system('python code/mpc/mpc_inference_orig.py --input_file {} --main_trie ./main.mpc --suffix_trie ./suffix.mpc --output_file ./completions.mpc'.format(os.path.join(args.data, 'unseen/test_formatted.txt')))
            os.system('python code/mpc/transform2eval.py --input_file ./completions.mpc --output_file {}'.format('out.' + output_map(os.path.join(args.data, 'unseen/test_formatted.txt')) +'.mpc.suffix'))
        else:
            # run commands
            # os.system('python code/mpc/create_train.py --inp {} --out ./train.mpc'.format(os.path.join(args.data, 'train.txt')))
            os.system('python code/mpc/main_trie_creation.py --input_file ./train.mpc --output_trie ./main.mpc --threshold 0')
            os.system('python code/mpc/mpc_inference_orig.py --input_file {} --main_trie ./main.mpc --output_file ./completions.mpc'.format(os.path.join(args.data, 'seen/test_formatted.txt')))
            os.system('python code/mpc/transform2eval.py --input_file ./completions.mpc --output_file {}'.format('out.' + output_map(os.path.join(args.data, 'seen/test_formatted.txt'))+ '.mpc'))
            os.system('python code/mpc/mpc_inference_orig.py --input_file {} --main_trie ./main.mpc --output_file ./completions.mpc'.format(os.path.join(args.data, 'unseen/test_formatted.txt')))
            os.system('python code/mpc/transform2eval.py --input_file ./completions.mpc --output_file {}'.format('out.' + output_map(os.path.join(args.data, 'unseen/test_formatted.txt')) + '.mpc'))
