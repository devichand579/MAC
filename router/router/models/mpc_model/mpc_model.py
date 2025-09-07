import sys
sys.path.append(".")
from router.utils.modelling_utils import BaseModel
from chatas.code.utils.dataset import Dialog, Utterance
import json


class MPCModel(BaseModel):
    def __init__(self, ckpt_dir: str = 'ckpts/mpc/MPC_MMDD/'):
        self.err = 0
        if not ckpt_dir.endswith('/'):
            ckpt_dir += '/'
        self.completions_json_path = ckpt_dir + 'completions.mpc.suffix'
        with open(self.completions_json_path, 'r') as f:
            # Load the completions from the JSON file
            lines = f.readlines()
            self.completions = {}
            for line in lines:
                # print(line)
                if line.strip() == "":
                    continue
                data = json.loads(line.strip())
                if 'id' in data and 'main_ppl' in data and 'suffix_ppl' in data:
                    self.completions[data['id']] = {
                        'main_ppl': data['main_ppl'] if data['main_ppl'] else 0.0,
                        'suffix_ppl': data['suffix_ppl'] if data['suffix_ppl'] else 0.0
                    }
        print(f"Loaded {len(self.completions)} completions from {self.completions_json_path}")
        self.signal_keys = ['idx', 'mpc_main_ppl', 'mpc_suffix_ppl']
        super().__init__()

    def get_signals(self, input_dialog: Dialog) -> dict[str, int | float | str]:
        """
        Extracts general features from the input dialog.

        :param input_dialog: An instance of Dialog containing the input data.
        :return: A dictionary with extracted features.
        """
        if not input_dialog.idx in self.completions:
            self.err+=1
            print(f"Warning: No completions found for dialog {input_dialog.idx}. Returning default values. {self.err} missing so far.")
            return {
                'idx': input_dialog.idx if input_dialog else 'unknown',
                'mpc_main_ppl': 0.0,
                'mpc_suffix_ppl': 0.0
            }
        return {
            'idx': input_dialog.idx,
            'mpc_main_ppl': self.completions[input_dialog.idx]['main_ppl'],
            'mpc_suffix_ppl': self.completions[input_dialog.idx]['suffix_ppl']
        }
        # assert input_dialog.level == 2, "only supports level 2 dialogs"
        # prefix = input_dialog.response.text
        # pred, cost = self.qb.Complete(prefix)[0][0][0]
        # # print(pred)
        # subword_len = self.qb.Complete(prefix)[0][0][1]
        # return {
        #     'idx': input_dialog.idx,
        #     'qb_nll': cost,
        #     'qb_pred_subword_len': subword_len
        # }
        
if __name__ == "__main__":
    model = MPCModel()
    # dialog = Dialog(idx="he__u2__s2", utterances = [Utterance(text="Hello", images=[]),
    #                                                 Utterance(text="How are you?", images=["image1.jpg"]),
    #                                                 Utterance(text="Ch", images=[])])
    # features = model.get_signals(dialog)
    # print(features)  # Output: {'idx': 'he__u2__s2', 'prefix_chars': 2, 'num_previous_utterance': 2, 'num_utterance_after_img': 1}