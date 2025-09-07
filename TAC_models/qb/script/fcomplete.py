import sys
sys.path.append('./')

from queryblazer import *
import io, argparse
import tqdm
import numpy as np




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch_factor', type=int, default=30)
    parser.add_argument('--beam_size', type=int, default=30)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--length_limit', type=int, default=100)
    parser.add_argument('--encoder', type=str, default="data/DDC/bpe/4096/encoder.fst")
    parser.add_argument('--model', type=str, default="data/DDC/bpe/4096/ngram.fst")
    parser.add_argument('--precomputed', type=str, default="data/DDC/bpe/4096/precomputed.bin")

    args = parser.parse_args()
    config = Config(branch_factor=args.branch_factor, beam_size=args.beam_size, topk=args.topk, length_limit=args.length_limit)
    qbz = QueryBlazer(encoder=args.encoder, model=args.model, config=config)
    assert qbz.LoadPrecomputed(args.precomputed)
    with io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if line=='\t'.join(['-']*2):
                print('\t'.join(['-']*5))
                continue
            x, y = line.split('\t')
            x = x.split("<eou>")[-1]
            # ----------------------------
            y = y.replace(" ,", ",")
            y = y.replace(" .", ".")
            y = y.replace(" ?", "?")
            y = y.replace(" !", "!")
            x = x.replace(" ,", ",")
            x = x.replace(" .", ".")
            x = x.replace(" ?", "?")
            x = x.replace(" !", "!")
            # ----------------------------
            predictions = np.array(qbz.Complete(x)[0])
            # print(predictions)
            preds = predictions[:, 0]
            subword_lens = predictions[:, 1]
            for ind, i in enumerate(preds):
                preds[ind] = (preds[ind][0], preds[ind][1] /subword_lens[ind])
            print('\t'.join([x, str([[i[0][len(x):], i[1]] for i in preds])]))


if __name__ == '__main__':
    main()
