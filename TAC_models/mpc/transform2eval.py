import json
import tqdm
import argparse

def create_outputs(inp, output):
    print("transforming outputs")
    c = 0
    with open(output, 'w') as wf:
        with open(inp, 'r') as f:
            for line in tqdm.tqdm(f):
                data = json.loads(line)
                prefix = data['prefix']
                gt = data['gt']
                pred = "-"
                confidence = "-"
                subword_len = "-"
                if(len(data['completions'])> 0):
                    pred = data['completions'][0][0][len(prefix):]
                    confidence = data['completions'][0][1]
                    subword_len = 1
                wf.write("\t".join([prefix, gt, pred, confidence, str(subword_len)]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='input file containing the predictions.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='output file for evaluation')
    args = parser.parse_args()
    create_outputs(args.input_file, args.output_file)