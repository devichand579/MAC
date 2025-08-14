import pandas as pd

from router.utils.modelling_utils import load_combined_pred_by_idx
from router.utils.dataset_utils import DatasetType, get_combined_score, get_dataset



if __name__ == '__main__':
    dataset = get_dataset(DatasetType.IMAGECHAT)
    for i in range(500):
        print(f"Dialog {i}:")
        print(dataset[i][0].format_dialog())
        print([utt.images for utt in dataset[i][0].utterances])
        print("\n---\n")