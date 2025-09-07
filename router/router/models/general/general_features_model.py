import sys
sys.path.append(".")
from router.utils.modelling_utils import BaseModel
from chatas.code.utils.dataset import Dialog, Utterance


class GeneralFeaturesModel(BaseModel):

    def __init__(self):
        """
        Initializes the GeneralFeaturesModel instance.
        Sets the signal_keys to include 'idx' and 'general_features'.
        """
        self.signal_keys = ['idx', 'prefix_chars', "num_previous_utterance", "num_utterance_after_img"]
        super().__init__()

    def get_signals(self, input_dialog: Dialog) -> dict[str, int | float | str]:
        """
        Extracts general features from the input dialog.

        :param input_dialog: An instance of Dialog containing the input data.
        :return: A dictionary with extracted features.
        """
        return {
            'idx': input_dialog.idx,
            'prefix_chars': len(input_dialog.response.text) if input_dialog.response.text else 0,
            'num_previous_utterance': len(input_dialog.utterances) - 1,
            'num_utterance_after_img': self._get_num_uttr_after_img(input_dialog)
        }

    def _get_num_uttr_after_img(self, input_dialog: Dialog) -> int:
        """
        Counts the number of utterances after the last image in the dialog.

        :param input_dialog: An instance of Dialog containing the input data.
        :return: The count of utterances after the last image.
        """
        if not input_dialog.utterances:
            return 0
        # Find the last utterance with images
        last_img_idx = next((i for i, utt in enumerate(reversed(input_dialog.utterances)) if utt.images), None)
        if last_img_idx is None:
            return len(input_dialog.utterances)
        return len(input_dialog.utterances) - last_img_idx - 1



if __name__ == "__main__":
    # Example usage
    model = GeneralFeaturesModel()
    dialog = Dialog(idx="he__u2__s2", utterances = [Utterance(text="Hello", images=[]),
                                                    Utterance(text="How are you?", images=["image1.jpg"]),
                                                    Utterance(text="Ch", images=[])])
    features = model.get_signals(dialog)
    print(features)  # Output: {'idx': 'he__u2__s2', 'prefix_chars': 2, 'num_previous_utterance': 2, 'num_utterance_after_img': 1}