import heapq
import re
import os

import math


def calculate_log_perplexity(probability_distribution):
    """
    Calculates the log of perplexity for a given probability distribution.

    Args:
        probability_distribution (list): A list of probabilities representing
                                         the distribution. The sum of probabilities
                                         should ideally be 1.

    Returns:
        float: The log of perplexity.
    """
    entropy = 0
    for prob in probability_distribution:
        if prob > 0:  # Ensure we don't take log of zero
            entropy -= prob * math.log2(prob)
        elif prob < 0:
            raise ValueError("Probabilities cannot be negative.")

    # Log of Perplexity = Entropy * log_e(2)
    log_perplexity = entropy * math.log(2)
    return log_perplexity


class TrieNode:
    def __init__(self):
        self.num_descendants = 0
        self.children = {}
        self.is_end_of_word = False

class QueryCompletionWithPerplexity:
    def _log_perplexity(self, node):
        if node.is_end_of_word:
            return 0
        vec = []
        for child in node.children.values():
            vec.append(child.num_descendants)
        if len(vec) == 0:
            return 0
        tot = sum(vec)
        if tot == 0:
            return 0
        prob_dist = [float(x) / tot for x in vec]
        log_perplexity = calculate_log_perplexity(prob_dist)
        return log_perplexity

    def __init__(self):
        self.root = TrieNode()
        self.query_frequency = {}

    def insert(self, query, frequency):
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node.num_descendants += 1
            node = node.children[char]
        node.is_end_of_word = True
        self.query_frequency[query] = frequency

    def _dfs(self, node, prefix, min_heap, k_completions):
        if node.is_end_of_word:
            query = prefix
            frequency = self.query_frequency[query]
            if len(min_heap) < k_completions:
                heapq.heappush(min_heap, (frequency, query))
            else:
                heapq.heappushpop(min_heap, (frequency, query))

        for char, child in node.children.items():
            self._dfs(child, prefix + char, min_heap, k_completions)

    def find_completions(self, prefix, k_completions=10):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return [], 0
            node = node.children[char]

        min_heap = []
        for char, child in node.children.items():
            self._dfs(child, prefix + char, min_heap, k_completions)
        completions = [(query, frequency) for frequency, query in sorted(min_heap, reverse=True)]
        return completions, self._log_perplexity(node)










class QueryCompletion:
    def __init__(self):
        self.root = TrieNode()
        self.query_frequency = {}

    def insert(self, query, frequency):
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node.num_descendants += 1
            node = node.children[char]
        node.is_end_of_word = True
        self.query_frequency[query] = frequency

    def _dfs(self, node, prefix, min_heap, k_completions):
        if node.is_end_of_word:
            query = prefix
            frequency = self.query_frequency[query]
            if len(min_heap) < k_completions:
                heapq.heappush(min_heap, (frequency, query))
            else:
                heapq.heappushpop(min_heap, (frequency, query))

        for char, child in node.children.items():
            self._dfs(child, prefix + char, min_heap, k_completions)

    def find_completions(self, prefix, k_completions=10):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        min_heap = []
        for char, child in node.children.items():
            self._dfs(child, prefix + char, min_heap, k_completions)
        completions = [(query, frequency) for frequency, query in sorted(min_heap, reverse=True)]
        return completions

def preprocess(input_str):
    input_str = input_str.strip().lower()
    # input_str = re.sub('\s+', ' ', input_str)
    return input_str

def create_suffixes(input_query):
    suffixes = []
    tokens = input_query.split(' ')
    if len(tokens)==0:
        return []
    temp_word = ''
    for token in tokens[::-1]:
        temp_word =  token + " " + temp_word
        suffixes.append(temp_word.strip())
    return suffixes[:-1]

def is_empty(input_text):
    input_text = input_text.strip()
    if input_text=='' or len(input_text)==0:
        return True
    return False

def load_text_stream(load_path, max_limit=-1, gt=0, context=0):
    if(gt):
        index = 0
        with open(load_path, 'r', encoding='utf-8') as f:
            for row in f:
                index+=1
                input_text = row.split('\t')[0].lstrip()
                if(context):
                    input_text = input_text.split('<eou>')[-1]  # Find the index of last <eou>
                    # input_text = input_text[last_eou_index + 5:]
                output_text = row.split('\t')[1].rstrip()
                if (max_limit>0 and index>max_limit):
                    yield False, "", ""
                yield True, input_text, output_text
        yield False, "", ""
    else:
        index = 0
        with open(load_path, 'r', encoding='utf-8') as f:
            for row in f:
                index+=1
                input_text = row.strip()
                if (max_limit>0 and index>max_limit):
                    yield False, "", ""
                yield True, input_text
        yield False, ""



def get_line_count(input_file, max_limit):
    # iterate over the input file and count the number of lines
    count=0
    stream = load_text_stream(input_file, max_limit)
    for status, _ in stream:
        if status==False:
            break
        count+=1
    return count

# check whether base_path to a file exists or not and create if it doesn't
def check_and_create_path(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))