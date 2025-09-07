import heapq
import re
import os
import pandas as pd

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class QueryCompletion:
    def __init__(self):
        self.root = TrieNode()
        self.query_frequency = {}

    def insert(self, query, frequency):
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
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
                    yield False, ""
                yield True, input_text
        yield False, ""



def load_text_streamMMD(load_path, max_limit=-1):
    index = 0
    df = pd.read_csv(load_path)
    for i in range(len(df)):
        input_text = df.iloc[i]['prefix']
        output_text = df.iloc[i]['suffix']
        id = df.iloc[i]['id']
        index+=1
        if (max_limit>0 and index>max_limit):
            yield False, "", "", ""
        yield True, id, input_text, output_text
    yield False, "", "", ""


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