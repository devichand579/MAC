from __future__ import annotations
from abc import abstractmethod
import argparse
from typing import Any, Dict, List, Callable
import pandas as pd
from tqdm import tqdm
import json
from code.utils.dataset import (
    TEST_CONFIG,
    DialogData,
    Dialog,
    MMDDData,
    create_image_path_by_url_mmdd,
)
from code.utils.misc import longest_Common_Prefix
from copy import deepcopy
import logging
from numpy import random

REQUIRED_COLUMNS = ["id", "pred"]
OPTIONAL_COLUMNS = ["images_used"]
METRICS = [
    "trigger_rate",
    "synctatic_match",
    "pr_recall",
    "pr_precision",
    "avg_pred_len",
    "tes",
    "dialog_support",
    "trigger_support",
]


class EvalDialog:
    dialog: Dialog
    idx: str
    sentence: str
    pred_by_id: Dict[str, str]
    images_used_by_id: Dict[str, int | None]
    match_by_id: Dict[str, int]
    p_precision_by_id: Dict[str, float]
    p_recall_by_id: Dict[str, float]
    pred_len_by_id: Dict[str, int]
    triggered_freq: int
    tes: float

    def __init__(self, dialog: Dialog):
        self.dialog = dialog
        assert (
            dialog.level == 1
        ), "dialog should be unrolled but not split (level 1)"
        self.idx = dialog.idx
        self.sentence = dialog.response.text
        self.pred_by_id = {}
        self.images_used_by_id = {}
        self.match_by_id = {}
        self.p_precision_by_id = {}
        self.p_recall_by_id = {}
        self.pred_len_by_id = {}
        self.triggered_freq = 0
        self.tes = 0

    def add_pred(self, idx: str, pred: str, images_used: int | None = None):
        assert idx.startswith(
            self.idx + "__s"
        ), f"pred idx {idx} does not match {self.idx}"
        if idx in self.pred_by_id:
            logging.warning(
                f"Duplicate pred idx {idx} in dialog {self.idx}. Overwriting"
            )
        if idx.endswith("__s0") or idx.endswith(f"__s{len(self.sentence)}"):
            return
        self.pred_by_id[idx] = pred
        self.images_used_by_id[idx] = images_used
        # if idx.startswith("te_bst:442__u5"):
        #     print(images_used)

    def _tes(self) -> float:
        """
        Calculates the effort saved by the user in typing the sentence
        """
        typed = 0
        effort = 0
        while typed < len(self.sentence):
            if (
                f"{self.idx}__s{typed}" in self.pred_by_id
                and self.pred_by_id[f"{self.idx}__s{typed}"]
            ):
                pred = self.pred_by_id[f"{self.idx}__s{typed}"]
                gt = self.sentence[typed:]
                if gt.startswith(pred):
                    typed += len(pred)
                else:
                    typed += 1
                    effort += 1
            else:
                typed += 1
                effort += 1
        return 1 - effort / len(self.sentence)

    def compute(self):
        self.triggered_freq = len(
            [pred for pred in self.pred_by_id.values() if pred]
        )
        self.match_by_id = {
            idx: int(pred == self.sentence[int(idx.split("__s")[1]) :])
            for idx, pred in self.pred_by_id.items()
            if pred
        }
        self.p_precision_by_id = {
            idx: len(
                longest_Common_Prefix(
                    [pred, self.sentence[int(idx.split("__s")[1]) :]]
                )
            )
            / len(pred)
            for idx, pred in self.pred_by_id.items()
            if pred
        }
        self.p_recall_by_id = {
            idx: len(
                longest_Common_Prefix(
                    [pred, self.sentence[int(idx.split("__s")[1]) :]]
                )
            )
            / len(self.sentence[int(idx.split("__s")[1]) :])
            for idx, pred in self.pred_by_id.items()
            if pred
        }
        self.pred_len_by_id = {
            idx: len(pred) for idx, pred in self.pred_by_id.items() if pred
        }
        self.tes = self._tes()

    @property
    def complete(self) -> bool:
        return (
            len(self.pred_by_id) == len(self.sentence) - 1
            and len(self.sentence) > 1
        )

    @property
    def images_used(self) -> int | None:
        unique_images_used = set(self.images_used_by_id.values())
        if len(unique_images_used) != 1:
            logging.warning(
                f"Multiple images used in dialog {self.idx}: {unique_images_used}"
            )
            return None
        return unique_images_used.pop()

    def __len__(self) -> int:
        return len(self.sentence) - 1


class EvalDialogData:
    dialogs_by_id: Dict[str, EvalDialog]
    dialogs: List[EvalDialog]

    def __init__(
        self,
        data: DialogData | None,
        preds: pd.DataFrame | None,
        adhoc_dialogs: List[EvalDialog] | None = None,
    ):
        if adhoc_dialogs:
            self.dialogs = adhoc_dialogs
            self.dialogs_by_id = {
                dialog.idx: dialog for dialog in adhoc_dialogs
            }
            return
        assert (
            data is not None and preds is not None
        ), "Either data or preds should be provided"

        preds = deepcopy(preds)
        preds = self.preprocess_preds(preds)

        if not all(col in preds.columns for col in REQUIRED_COLUMNS):
            raise ValueError(
                f"Missing required columns in preds: {REQUIRED_COLUMNS}"
            )
        for col in OPTIONAL_COLUMNS:
            if col in preds.columns:
                logging.warning(f"Optional column {col} found in preds")
        assert (
            data[0][0].level == 1
        ), "data should be unrolled but not split (level 1)"

        self.dialogs_by_id = {}
        for dialog, _ in data:
            self.dialogs_by_id[dialog.idx] = EvalDialog(dialog)
        for _, row in tqdm(
            preds.iterrows(), total=len(preds), desc="Assigning preds"
        ):
            dialog_idx = row["id"].split("__s")[0]
            if dialog_idx not in self.dialogs_by_id:
                logging.warning(
                    f"Dialog idx {dialog_idx} not found in data, but found in preds ... skipping"
                )
                continue
            self.dialogs_by_id[dialog_idx].add_pred(
                row["id"], row["pred"], row.get("images_used", None)
            )
        self.dialogs_by_id = {
            dialog.idx: dialog
            for dialog in self.dialogs_by_id.values()
            if dialog.complete
        }
        self.dialogs = list(self.dialogs_by_id.values())
        if len(self.dialogs) < len(self.dialogs_by_id):
            logging.warning(
                f"Only {len(self.dialogs)} dialogs are complete out of {len(self.dialogs_by_id)}"
            )

    def seen_unseen_split(self, intersection: Dict[str, int]):
        """
        Split the data into seen and unseen based on the intersection.
        """
        seen = []
        unseen = []
        for dialog in self.dialogs:
            if dialog.idx in intersection and intersection[dialog.idx]:
                seen.append(dialog)
            else:
                unseen.append(dialog)

        # Create new EvalDialogData instances using adhoc_dialogs
        seen_data = EvalDialogData(None, None, adhoc_dialogs=seen)

        total_seen = 0
        for dialog in seen:
            assert type(dialog) == EvalDialog
            total_seen += len(dialog.dialog.create_splits())
        print("Total seen dialogs:", len(seen), "Total seen instances:", total_seen)
        # exit(0)

        unseen_data = EvalDialogData(None, None, adhoc_dialogs=unseen)
        return seen_data, unseen_data

    def prune(self, fn: Callable[[EvalDialog], bool]) -> EvalDialogData:
        """
        Prune the dialogs based on the function provided.

        Args:
            fn: A function that takes a eval_dialog and returns a boolean value.

        Returns:
            A new EvalDialogData instance with pruned dialogs.
        """
        pruned_dialogs = [
            eval_dialog for eval_dialog in self.dialogs if fn(eval_dialog)
        ]
        return EvalDialogData(None, None, adhoc_dialogs=pruned_dialogs)

    def compute(self) -> Dict[str, Any]:
        matches = []
        p_precisions = []
        p_recalls = []
        pred_lens = []
        triggered_freq = 0
        tess = []
        total_instances = 0
        for dialog in tqdm(self.dialogs, desc="Computing metrics"):
            dialog.compute()
            total_instances += len(dialog)
            p_precisions.extend(dialog.p_precision_by_id.values())
            p_recalls.extend(dialog.p_recall_by_id.values())
            matches.extend(dialog.match_by_id.values())
            pred_lens.extend(dialog.pred_len_by_id.values())
            triggered_freq += dialog.triggered_freq
            tess.append(dialog.tes)
        assert (
            len(matches)
            == len(p_precisions)
            == len(p_recalls)
            == len(pred_lens)
            == triggered_freq
        ), f"Length mismatch: {len(matches)}, {len(p_precisions)}, {len(p_recalls)}, {len(pred_lens)}, {triggered_freq}"

        assert len(tess) == len(
            self.dialogs
        ), f"Length mismatch: {len(tess)}, {len(self.dialogs)}"

        metrics = {
            "trigger_rate": triggered_freq / total_instances,
            "synctatic_match": sum(matches) / triggered_freq,
            "pr_precision": sum(p_precisions) / triggered_freq,
            "pr_recall": sum(p_recalls) / triggered_freq,
            "avg_pred_len": sum(pred_lens) / triggered_freq,
            "tes": sum(tess) / len(tess),
            "dialog_support": len(self.dialogs),
            "instances": total_instances,
            "trigger_support": triggered_freq,
        }
        return metrics

    @abstractmethod
    def preprocess_preds(self, preds: pd.DataFrame) -> pd.DataFrame:
        """
        can be overridden later if required
        """
        # fillna with empty string
        preds = preds.fillna("")
        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="its the outfile"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/MMDD/test.csv",
        help="data/MMDD/test.csv",
    )
    parser.add_argument(
        "--intersection_path",
        type=str,
        default=None,
        help="data/DialogCC/intersection.json",
    )
    parser.add_argument(
        "--jai",
        action="store_true",
        help="use just after image data",
    )
    parser.add_argument(
        "--sample", action="store_true", help="use sample data"
    )
    # parser.add_argument(
    #     "--output", type=str, required=True, help="its the desired csv file"
    # )
    args = parser.parse_args()
    preds = pd.read_csv(args.input)
    # print(df.head(5))
    TEST_CONFIG["to_split"] = False
    if args.sample:
        TEST_CONFIG["sample"] = 30
    ## ADHOC CODE
    test_data = MMDDData(
        args.data_path,
        image_path_by_url=create_image_path_by_url_mmdd(
            "data/MMDD/images/"
        ),
        **TEST_CONFIG,
    )
    eval_data = EvalDialogData(test_data, preds)
    if args.jai:
        eval_data = eval_data.prune(
            lambda eval_dialog: (
                True if eval_dialog.dialog[-1].images else False
            )
        )

    if args.intersection_path:
        with open(args.intersection_path, "r") as f:
            intersection = json.load(f)["test_key"]
        seen, unseen = eval_data.seen_unseen_split(intersection)
        seen_metrics = seen.compute()
        unseen_metrics = unseen.compute()
        print("Seen metrics:", seen_metrics)
        print("Unseen metrics:", unseen_metrics)

    metrics = eval_data.compute()
    print("ALL Metrics:", metrics)


    # if "images_used" in preds.columns:
    #     unique_images_used = preds["images_used"].unique()
    #     for image_used in sorted(unique_images_used):
    #         dialogs_here = [
    #             i for i in eval_data.dialogs if i.images_used == image_used
    #         ]
    #         metrics = EvalDialogData(
    #             None, None, adhoc_dialogs=dialogs_here
    #         ).compute()
    #         print(f"Metrics for image_used={image_used}: {metrics}")
