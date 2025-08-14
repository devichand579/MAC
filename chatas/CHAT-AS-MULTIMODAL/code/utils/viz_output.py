import pandas as pd
import argparse
import re
from dataset import (
    DialogCCData,
    DialogData,
    TEST_CONFIG,
    create_image_path_by_url,
)


DATASET_CLASS_BY_NAME = {
    "dialogcc": DialogCCData,
}


def get_dataset(
    args: argparse.Namespace, to_split: bool, to_replace: bool
) -> DialogData:
    """Get the dataset object"""
    return DATASET_CLASS_BY_NAME[args.dataset_class](
        path=args.dataset_path,
        image_path_by_url=create_image_path_by_url(
            args.image_names_path, args.images_path
        ),
        **{**TEST_CONFIG, "to_split": to_split, "to_replace": to_replace},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--dataset_class", type=str, default="dialogcc")
    parser.add_argument(
        "--dataset_path", type=str, default="data/DialogCC/test.csv"
    )
    parser.add_argument(
        "--image_names_path", type=str, default="../tmp/image_names"
    )
    parser.add_argument("--images_path", type=str, default="../tmp/images_n")
    args = parser.parse_args()

    # assert that the id is of the form te_bst:517__u3__s32
    assert re.match(r"^\w+:\d+__u\d+$", args.id)

    dataset = get_dataset(args, to_split=False, to_replace=True)
    dialog, _ = dataset.dialog_suffix_by_id[args.id]
    splits = dialog.create_splits()
    df = pd.read_csv(args.outfile)
    # assert  {"id", "pred"} in set(df.columns), "Invalid columns"
    df = df[df["id"].apply(lambda x: x.startswith(args.id + "__s"))]
    print(f"Found {len(df)} predictions for ID `{args.id}`")
    assert len(df) != 0, f"ID `{args.id}` not found in the output file"
    pred_by_id = {row["id"]: row["pred"] for _, row in df.iterrows()}

    # print the dialog
    print("Dialog:")
    print(dialog.format_dialog())
    print("Predictions:")
    for split, suff in splits:
        print(f"{split.response.text}\t{suff}\t{pred_by_id.get(split.idx, '')}")
