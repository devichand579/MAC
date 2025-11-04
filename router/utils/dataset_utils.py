import sys

from router.utils.modelling_utils import CombinedPred, save_combined_pred_by_idx

sys.path.append(".")
from enum import Enum
from router.models.main import Model
import os
import logging
import pandas as pd
from tqdm import tqdm
from chatas.code.utils.dataset import (
    DialogData,
    MMDDData,
    ImageChatData,
    create_image_path_by_url_mmdd,
    create_image_path_by_url_image_chat,
)
from chatas.eval import longest_Common_Prefix

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

import numpy as np
np.random.seed(42)

class DatasetType(Enum):
    MMDD = "mmdd"
    IMAGECHAT = "imagechat"


def partial_precision(pred: str, gt: str) -> float:
    # print("partial_precision ", pred, gt)
    if not pred:
        return 1.0
    prefix = longest_Common_Prefix([pred, gt])
    return len(prefix) / len(pred)

def synctatic_match(pred: str, gt: str) -> float:
    """
    Computes the syntactic match score between the predicted and ground truth strings.
    
    :param pred: The predicted string.
    :param gt: The ground truth string.
    :return: A float representing the syntactic match score.
    """
    if not pred or not gt:
        return 0.0
    return 1.0 if pred == gt else 0.0

def partial_recall(pred: str, gt: str) -> float:
    if not gt:
        return 1.0
    prefix = longest_Common_Prefix([pred, gt])
    return len(prefix) / len(gt)


def partial_f1(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    
    precision = partial_precision(pred, gt)
    recall = partial_recall(pred, gt)

    # print(f"{pred=}, {gt=}, {precision=}, {recall=}")
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def get_combined_score(
    dataset: DialogData,
    combined_pred_by_idx: dict[str, CombinedPred],
    y_idx_pred_model: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculates the combined score for the predictions based on the dataset and combined predictions.

    :param dataset: An instance of DialogData containing the dataset.
    :param combined_pred_by_idx: A dictionary mapping indices to CombinedPred instances.
    :param y_idx_pred_model: A DataFrame containing predictions by model.
    :return: The combined score as a float.
    """
    total_f1_score = 0.0
    total_pr_score = 0.0
    total_re_score = 0.0
    total_sm_score = 0.0
    count = 0
    for idx, row in y_idx_pred_model.iterrows():
        dialog, gt = dataset.dialog_suffix_by_id[row["idx"]]
        if dialog is None:
            raise ValueError(
                f"Dialog for index {row['idx']} not found in dataset."
            )
        combined_pred = combined_pred_by_idx.get(row["idx"])
        if not combined_pred:
            raise ValueError(
                f"Combined prediction for index {row['idx']} not found."
            )
        pred = combined_pred.pred_by_model.get(
            row["model"], ""
        )
        # gt = dialog.response.text
        if not isinstance(gt, str):
            logger.warning(
                f"Expected gt to be a string, got {type(gt)} for idx {row['idx']}"
            )
            gt = str(gt)  # Ensure gt is a string
        # assert isinstance(
        #     gt, str
        # ), f"Expected gt to be a string, got {type(gt)} for idx {row['idx']}"
        # logger.info(
        #     f"{pred=}, {gt=}, {row['model']=}, {idx=}"
        # )
        f1_score = partial_f1(pred, gt)
        pr_score = partial_precision(pred, gt)
        re_score = partial_recall(pred, gt)
        total_f1_score += f1_score
        total_pr_score += pr_score
        total_re_score += re_score
        sm_score = synctatic_match(pred, gt)
        total_sm_score += sm_score
        count += 1
    

    scores = {
        "f1": total_f1_score / count if count > 0 else 0.0,
        "precision": total_pr_score / count if count > 0 else 0.0,
        "recall": total_re_score / count if count > 0 else 0.0,
        "syntactic_match": total_sm_score / count if count > 0 else 0.0,
    }

    return scores


def get_dataset(dataset: DatasetType) -> DialogData:
    """
    Returns the dataset based on the provided dataset type.

    :param dataset: An instance of DatasetType enum.
    :return: An instance of DialogData containing the dataset.
    """
    if dataset == DatasetType.MMDD:
        test_data = MMDDData(
            path="data/MMDD/test.csv",
            to_filter=True,
            to_replace=False,
            image_path_by_url=create_image_path_by_url_mmdd("data/MMDD/images"),
            to_unroll=True,
            min_images_per_dialog=1,
            to_split=True,
        )
        dialogs = [
            dialog
            for dialog, suff in test_data
            if not dialog.utterances[-1].images
        ]
        suffixes = [
            suff
            for dialog, suff in test_data
            if not dialog.utterances[-1].images
        ]
        print(len(dialogs))
        test_data = DialogData(
            path=(dialogs, suffixes),
        )
        return test_data
    elif dataset == DatasetType.IMAGECHAT:
        test_data = ImageChatData(
            path="data/ImageChat/image_chat/test.csv",
            to_filter=True,
            to_replace=True,
            image_path_by_url=create_image_path_by_url_image_chat(
                "data/ImageChat/yfcc_images"
            ),
            to_unroll=True,
            min_images_per_dialog=1,
            n_samples=4800,
            to_split=True,
        )
        dialog = test_data.dialogs
        suffixes = test_data.suffixes
        return test_data
    else:
        raise NotImplementedError(
            f"Dataset {dataset} is not implemented yet."
        )

def get_combined_pred_by_idx(
    dataset_type: DatasetType,
) -> dict[str, CombinedPred]:
    """
    Creates a dictionary mapping indices to CombinedPred instances based on the dataset type.

    :param dataset_type: An instance of DatasetType enum.
    :return: A dictionary mapping indices to CombinedPred instances.
    """
    data = get_dataset(dataset_type)
    print(data[0][0].format_dialog())
    outfiles = []
    if dataset_type == DatasetType.MMDD:
        for model in Model:
            outfile = f"./mmdd_outputs/mmdd.{model.value}.csv"
            if not os.path.exists(outfile):
                logger.warning(
                    f"file {outfile} does not exist."
                )
                continue
            print(outfile)
            outfiles.append((outfile, model.value))
    else:
        for model in Model:
            outfile = f"./imagechat_outputs/imagechat.{model.value}.csv"
            if not os.path.exists(outfile):

                logger.warning(
                    f"file {outfile} does not exist."
                )
                continue
            outfiles.append((outfile, model.value))
    outdfs = [
        (pd.read_csv(outfile), val)
        for outfile, val in outfiles
    ]
    outdfs = [
        (outdf.fillna(""), val) for outdf, val in outdfs
    ]  # fill NaN with empty string
    out_dicts = [
        (outdf.set_index("id")["pred"].to_dict(), model)
        for outdf, model in outdfs
    ]
    outids = [
        (outdf["id"].tolist(), val) for outdf, val in outdfs
    ]
    # intersection of all ids
    common_ids = set([dialog.idx for dialog, _ in data])
    print("total ids in data:", len(common_ids))
    for ids, val in outids:
        common_ids.intersection_update(ids)
    common_ids = list(common_ids)
    common_ids.sort()
    print(
        "after intersection, common ids:", len(common_ids)
    )
    suffixes = [
        (
            str(data.dialog_suffix_by_id[id][1])
            if data.dialog_suffix_by_id[id][1]
            else ""
        )
        for id in common_ids
    ]
    combined_pred_by_idx = {}
    for idx, id in tqdm(
        enumerate(common_ids),
        total=len(common_ids),
        desc="Creating CombinedPreds",
    ):
        pred_model_list = [
            (out_dict.get(id, ""), model)
            for out_dict, model in out_dicts
        ]
        combined_pred_by_idx[str(id)] = CombinedPred(
            idx=str(id),
            pred_by_model={
                model: pred
                for pred, model in pred_model_list
            },
        )
    return combined_pred_by_idx


def process_combined_pred(
    dataset: DialogData,
    combined_pred: CombinedPred,
) -> dict[str, str|float]:
    """
    Processes a CombinedPred instance to extract predictions by model.

    :param combined_pred: An instance of CombinedPred.
    :return: A dictionary mapping model names to their predictions.
    """
    preds_models = [(pred, model) for model, pred in combined_pred.pred_by_model.items()]
    score_pred_models = [
        (
            partial_f1(pred, str(dataset.dialog_suffix_by_id[combined_pred.idx][1])),
            pred,
            model,
        )
        for pred, model in preds_models
    ]
    best_score_pred_model = max(score_pred_models, key=lambda x: x[0])
    best_model = best_score_pred_model[2]
    best_pred = best_score_pred_model[1]
    best_score = best_score_pred_model[0]
    return {
        "pred": best_pred,
        "model": best_model,
        "score": best_score,
    }


def get_master_dataset(dataset_type: DatasetType) -> str:
    """
    Returns the master dataset name based on the provided dataset type.

    :param dataset: An instance of DatasetType enum.
    :return: The name of the master dataset.
    """
    combined_pred_by_idx = get_combined_pred_by_idx(
        dataset_type
    )
    dataset = get_dataset(dataset_type)
    print(dataset[0][0].__repr__())
    save_combined_pred_by_idx(combined_pred_by_idx, "combined_pred_by_idx.json")
    logger.info("Combined predictions saved to combined_pred_by_idx.json")

    master_data = [
        {
            "idx": idx,
            **process_combined_pred(
                dataset, combined_pred
            ),
        }
        for idx, combined_pred in tqdm(
            combined_pred_by_idx.items(),
            desc="Processing Combined Predictions",
        )
    ]
    df = pd.DataFrame(master_data)
    df.to_csv("master_dataset.csv", index=False)
    logger.info("Master dataset created and saved to master_dataset.csv")
    return "master_dataset.csv"


if __name__ == "__main__":
    dataset = DatasetType.MMDD
    master_dataset = get_master_dataset(dataset)
    logger.info(f"Master dataset created: {master_dataset}")
    
