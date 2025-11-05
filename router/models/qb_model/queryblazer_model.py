import sys
sys.path.append(".")
from router.utils.modelling_utils import BaseModel
from chatas.code.utils.dataset import Dialog, Utterance
from queryblazer import QueryBlazer, Config # type: ignore
import argparse

class Args:
    def __init__(self, ckpt_dir: str = 'QB_ckpts/QB_MMDD'):
        # Output Files (will overwrite)
        if not ckpt_dir.endswith('/'):
            ckpt_dir += '/'
        self.OUTPUT_DIR = ckpt_dir
        self.LOG_ENCODED = self.OUTPUT_DIR + 'train.enc'
        self.SPM_PREFIX = self.OUTPUT_DIR + 'subword'  # SPM_PREFIX.{m, vocab}
        self.ENCODER = self.OUTPUT_DIR + 'encoder.fst'
        self.LANGUAGE_MODEL = self.OUTPUT_DIR + 'ngram.fst'  # LANGUAGE_MODEL.{arpa, fst}
        self.PRECOMPUTED = self.OUTPUT_DIR + 'precomputed.bin'

        # Config
        self.SPM_MODEL = 'bpe'  # char, bpe, unigram
        self.SPM_VOCAB_SIZE = 4096
        self.SPM_CHARACTER_COVERAGE = 0.9995
        self.LM_ORDER = 8
        self.LM_PRUNE = "--prune 0 1 1 2 2 3 3 4"

        self.BRANCH_FACTOR = 30
        self.BEAM_SIZE = 30
        self.TOPK = 10
        self.LENGTH_LIMIT = 100


class QueryBlazerModel(BaseModel):
    def __init__(self, ckpt_dir: str = 'QB_ckpts/QB_MMDD'):
        """
        Initializes the QueryBlazerModel instance.
        Sets the signal_keys to include 'idx', 'prefix_chars', 'num_previous_utterance', and 'num_utterance_after_img'.
        """
        self.signal_keys = ['idx', 'qb_nll']
        self.args = Args(ckpt_dir=ckpt_dir)
        self.qb = QueryBlazer(encoder=self.args.ENCODER,
                              model=self.args.LANGUAGE_MODEL,
                              config=Config(
                                  branch_factor=self.args.BRANCH_FACTOR,
                                  beam_size=self.args.BEAM_SIZE,
                                  topk=self.args.TOPK,
                                  length_limit=self.args.LENGTH_LIMIT
                              ))
        super().__init__()

    def get_signals(self, input_dialog: Dialog) -> dict[str, int | float | str]:
        """
        Extracts general features from the input dialog.

        :param input_dialog: An instance of Dialog containing the input data.
        :return: A dictionary with extracted features.
        """
        assert input_dialog.level == 2, "only supports level 2 dialogs"
        prefix = input_dialog.response.text
        pred, cost = self.qb.Complete(prefix)[0][0][0]
        # print(pred)
        subword_len = self.qb.Complete(prefix)[0][0][1]
        return {
            'idx': input_dialog.idx,
            'qb_nll': cost,
            'qb_pred_subword_len': subword_len
        }
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='QB_ckpts/QB_MMDD')
    args = parser.parse_args()
    model = QueryBlazerModel(ckpt_dir=args.ckpt_dir)
    dialog = Dialog(idx="he__u2__s2", utterances = [Utterance(text="Hello", images=[]),
                                                    Utterance(text="How are you?", images=["image1.jpg"]),
                                                    Utterance(text="Ch", images=[])])
    features = model.get_signals(dialog)
    print(features)  # Output: {'idx': 'he__u2__s2', 'prefix_chars': 2, 'num_previous_utterance': 2, 'num_utterance_after_img': 1}