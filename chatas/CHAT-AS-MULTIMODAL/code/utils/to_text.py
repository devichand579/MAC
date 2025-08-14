""" Convert the dataset to text format """

from dataset import DialogCCData, Dialog, Utterance, create_image_path_by_url
from typing import List
import argparse
import os
import pandas as pd

def dialog2row(
    args: argparse.Namespace, dialog: Dialog, suffix: str | None
) -> List[str]:
    """Convert the last line of dialog to a row"""
    row = []
    if args.ids:
        row.append(f"{dialog.idx}")
    row.append(dialog.response.text)
    if args.to_split:
        row.append(suffix)
    return row


def save_pandas(data: List[List[str]], output_path: str):
    """Save the data to a pandas dataframe"""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/DialogCC/train.csv"
    )
    parser.add_argument(
        "--output_path", type=str, default="junk/MPC/DialogCC_train.txt"
    )
    parser.add_argument(
        "--to_split", action="store_true", help="Split the dialog into turns"
    )
    parser.add_argument(
        "--ids", action="store_true", help="Print dialog ids too"
    )
    parser.add_argument(
        "--no_pandas", action="store_true", help="Save as a text file"
    )
    args = parser.parse_args()
    dialog_data = DialogCCData(
        args.data_path,
        to_filter=True,
        to_replace=False,
        image_path_by_url=create_image_path_by_url(
            "../tmp/image_names", "../tmp/images_n"
        ),
        to_unroll=True,
        min_images_per_dialog=1,
        n_samples=1100,
        to_split=args.to_split,
    )
    data = []
    for id in range(len(dialog_data)):
        data.append(dialog2row(args, *dialog_data[id]))

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    if args.no_pandas:
        with open(args.output_path, "w") as f:
            for row in data:
                f.write("\t".join(row) + "\n")
    else:
        save_pandas(data, args.output_path)

    print(f"Saved to {args.output_path}")
