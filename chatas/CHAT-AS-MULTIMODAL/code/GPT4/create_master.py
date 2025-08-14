import sys
import os
sys.path = [(os.path.join(os.path.dirname(__file__), "../../"))] + sys.path
from code.utils.dataset import DialogData, Dialog, DialogCCData, create_image_path_by_url
from typing import List, Dict
from tqdm import tqdm
import pandas as pd


REQUIRED_COLUMNS = ["id", "context", "prefix"]
K = None  # Number of rows to sample for the master dataset
# K = 30
seed = 42


def preprocess(dialog: Dialog) -> Dialog:
    """
    GPT4 specific preprocess dialog object for GPT4.

    Args:
        dialog (Dialog): Dialog object

    Returns:
        Dialog: Preprocessed dialog object
    """
    # use escaping for Quotes
    for utterance in dialog:
        pass
        # utterance.text = utterance.text.replace('"', '\\"')
        # utterance.text = utterance.text.replace("'", "\\'")
    return dialog

def format_row(dialog: Dialog) -> Dict[str, str]:
    """
    Format a row for the master dataset.

    Args:
        dialog (Dialog): Dialog object

    Returns:
        Dict[str, str]: Formatted row
    """
    dialog = preprocess(dialog)
    row = {}
    row["id"] = dialog.idx
    context_utterances = dialog.context
    prefix_utterance = dialog.response
    context = ""
    for idx, utterance in enumerate(context_utterances):
        if utterance.images:
            images_str = "\\n".join([f"IMAGE:{image}" for image in utterance.images])
            context += (
                f"{utterance.speaker}:\\n{images_str}\\n{utterance.text}\\n"
            )
        else:
            context += f"{utterance.speaker}:\\n{utterance.text}\\n"
    row["context"] = context
    prefix = ""
    if prefix_utterance.images:
        images_str = "\\n".join([f"IMAGE:{image}" for image in prefix_utterance.images])
        prefix += f"{prefix_utterance.speaker}:\\n{images_str}\\n{prefix_utterance.text}"
    else:
        prefix += f"{prefix_utterance.speaker}:\\n{prefix_utterance.text}"
    row["prefix"] = prefix
    return row


def write_dataframe_to_file(df: pd.DataFrame, file_path: str):
    """
    Write a dataframe to a file.

    Args:
        df (pd.DataFrame): Dataframe
        file_path (str): File path
    """
    with open(file_path, "w") as f:
        f.write("\t".join(REQUIRED_COLUMNS) + "\n")
        for idx, row in df.iterrows():
            f.write(f"{row['id']}\t{row['context']}\t{row['prefix']}\n")
    with open(file_path, "r") as f:
        n_lines_saved = len(f.readlines())
    assert len(df) == n_lines_saved - 1


def create_master(dataset: DialogData) -> List[List[str]]:
    """
    Create master dataset by unrolling all dialogs in the dataset.

    Args:
        dataset (DialogData): Dataset object containing dialogs

    Returns:
        List[List[str]]: List of dialogs in the format of a list of strings
    """
    formatted = []
    for i in tqdm(range(len(dataset))):
        dialog, _ = dataset[i]
        formatted_row_dict = format_row(dialog)
        formatted.append([formatted_row_dict[col] for col in REQUIRED_COLUMNS])
    return formatted


if __name__ == "__main__":
    dialog_data = DialogCCData(
        path="data/DialogCC/test.csv",
        to_filter=True,
        to_replace=False,
        image_path_by_url=create_image_path_by_url(
            "../tmp/image_names", "../tmp/images_n"
        ),
        to_unroll=True,
        min_images_per_dialog=1,
        n_samples=1100 if not K else K,
        to_split=True,
    )
    master = create_master(dialog_data)
    df = pd.DataFrame(master, columns=REQUIRED_COLUMNS)
    print("size of master dataset:", len(df))
    if not K:
        # df.to_csv("junk/GPT4/DialogCC/master.txt", index=False, sep="\t")
        write_dataframe_to_file(df, "junk/GPT4/DialogCC/master.txt")
    else:
        # df.to_csv("junk/GPT4/DialogCC/sample_master.txt", index=False, sep="\t")
        write_dataframe_to_file(df, "junk/GPT4/DialogCC/sample_master.txt")
