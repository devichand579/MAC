import argparse
from utils import QueryCompletion
import pickle

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--suffix_trie", type=str, default = "./suffix.mpc")
    args = args.parse_args()
    # find the number of nodes in the trie
    with open(args.suffix_trie, 'rb') as f:
        trie = pickle.load(f)
    print("coverage = ", len(trie.query_frequency))
