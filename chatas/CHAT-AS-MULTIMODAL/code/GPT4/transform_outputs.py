"""Transforms the output of the GPT-4 model into our format"""

import argparse
import pandas as pd
import os
import sys
import logging
from termcolor import colored

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataset import (
    DialogCCData,
    DialogData,
    Dialog,
    TEST_CONFIG,
    create_image_path_by_url,
)


APPROX_PREFIX_MATCH_THRESHOLD = 0.5
APPROX_PREFIX_MATCH_BOUNDARY_SPAN = 1


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def preprocess_result(result: str, dialog: Dialog) -> str:
    """Preprocesses the result"""
    # print(result)
    result = result.lower()  # we can strip whole sentence
    # print(result)
    result = result.replace("<result>", "").replace("</result>", "")
    split_result = result.split("\\n")
    split_result = list(filter(lambda x: "image:http" not in x, split_result))
    result = "\\n".join(split_result)
    split_result = result.split("||")
    result = split_result[-1]
    result = result.replace("\\n", "")
    result = result.lstrip("<").rstrip(">")
    result = result.replace("prefix:", "").replace("prefix :", "")
    result = result.strip()
    if dialog.response.speaker:
        speaker = dialog.response.speaker.lower()
        # print(speaker)
        if result.startswith(f"{speaker}:"):
            result = result[len(f"{speaker}:") :].strip()
        if result.startswith(f"{speaker} :"):
            result = result[len(f"{speaker} :") :].strip()
    return result


def prefix_match(prefix: str, result: str) -> bool:
    """Checks if the result matches the prefix"""
    return result.startswith(prefix)


def approx_prefix_match(prefix: str, result: str) -> bool:
    """Checks if the result approximately matches the prefix"""
    if not prefix.split():
        return True
    prefix_words = prefix.split()
    result_words = result.split()
    result_prefix_words = result_words[
        : len(prefix_words) + APPROX_PREFIX_MATCH_BOUNDARY_SPAN
    ]
    return (
        len(set(prefix_words) & set(result_prefix_words)) / len(set(prefix_words))
        > APPROX_PREFIX_MATCH_THRESHOLD
    )


def get_max_match_len(prefix: str, result: str) -> int:
    """Gets the maximum length of the prefix that matches the result"""
    for i in range(len(prefix), 0, -1):
        if result.startswith(prefix[:i]):
            return i
    return 0


def transform(row: pd.Series, data: DialogData) -> pd.Series:
    """Transforms a row of the output dataframe into our format"""
    result = row["Result.OutputResult"]
    idx = row["id"]
    if idx in data.dialog_suffix_by_id:
        dialog, _ = data.dialog_suffix_by_id[idx]
        dialog = DialogData.preprocess(dialog)
    else:
        raise ValueError(
            f"Dialog id found in predictions but not found in test data: {idx}"
        )
    result = preprocess_result(result, dialog)
    prefix = dialog.response.text
    pred = ""
    if prefix_match(prefix, result):
        pred = result[len(prefix) :]
    elif not approx_prefix_match(prefix, result):
        # Now assume that the model directly predicted the continuation
        # check if a prefix of result matches a suffix of prefix
        for l in range(len(result), 0, -1):
            if prefix.endswith(result[:l]):
                pred = result[l:]
                break
        # if no prefix of result matches a suffix of prefix, then assume that the model started from new word
        if not pred:
            pred = result
    else:
        match_len = get_max_match_len(prefix, result)
        logger.warn(
            f"Prefix does not match result for dialog id '{idx}': '{colored(prefix[:match_len], 'green')}{colored(prefix[match_len:], 'red')}' vs '{colored(result[:match_len], 'green')}{colored(result[match_len:], 'red')}'"
        )
    new_row = pd.Series(
        {
            "id": idx,
            "pred": pred,
        }
    )
    return new_row


def read_tsv(path: str) -> pd.DataFrame | None:
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if len(line) != len(lines[0]):
            logger.warn(
                f"Line {i} has different number of columns in {path}, ",
            )
            return None
    df = pd.DataFrame(lines[1:], columns=lines[0])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        # required=True,
        default="./29Nov2024-GPT4_no_esc/outputs/",
        help="Path to the directory containing the outputs",
    )
    parser.add_argument("--output_path", type=str, default="out.all.dcc.gpt4_")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    files = list(
        filter(lambda f: f.endswith(".tsv"), os.listdir(args.input_dir))
    )
    dfs = [
        read_tsv(
            os.path.join(args.input_dir, f),
        )
        for f in files
    ]
    dfs = list(filter(lambda x: x is not None, dfs))
    df = pd.concat(dfs, ignore_index=True)
    df.fillna("", inplace=True)
    if args.sample:
        TEST_CONFIG["n_samples"] = 30
    data = DialogCCData(
        path="data/DialogCC/test.csv",
        image_path_by_url=create_image_path_by_url(
            "../tmp/image_names", "../tmp/images_n"
        ),
        **TEST_CONFIG,
    )
    new_df_data = []
    empty_count = 0
    total_count = 0
    for idx, row in df.iterrows():
        total_count += 1
        row = transform(row, data)
        if not row["pred"]:
            empty_count += 1
            progress = total_count / len(df)
            empty_percentage = empty_count / total_count if total_count > 0 else 0

            # Print the formatted output
            sys.stdout.write(f"\rCompleted: {progress * 100:.2f}% | Empty: {empty_percentage * 100:.2f}%")
            sys.stdout.flush()
    new_df = df.apply(lambda row: transform(row, data), axis=1)
    new_df.to_csv(args.output_path, index=False)
