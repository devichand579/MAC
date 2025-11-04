# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import List, Literal

from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from chatas directly
from chatas.code.utils.dataset import (
    Dialog,
    MMDDData,
    create_image_path_by_url_mmdd,
    ImageChatData,
    create_image_path_by_url_image_chat,
)
import pandas as pd
from swift.llm import (
    InferEngine,
    InferRequest,
    PtEngine,
    RequestConfig,
)
from argparse import ArgumentParser

import torch


# Defaults; can be overridden by CLI flags
OBS = 128         
BS = 64          
OUTPUT_DIR = "../output/"
SPLIT = 0       
OUTPUT_FILE = "output.csv"  

NO_IMG = False

def transform_dialog_data_to_message(dialog: Dialog, suffix: str) -> dict[str, any]:
    query = ""
    images = []
    for utterance in dialog.utterances[:-1]:
        if len(utterance.images) > 0:
            if not NO_IMG:
                images.extend(utterance.images)
            query += f"{utterance.speaker}:<|IMAGE|>\n"
        else:
            query += f"{utterance.speaker}:{utterance.text}\n"
    images.extend(dialog.utterances[-1].images)
    query += f"{dialog.utterances[-1].speaker}:{dialog.utterances[-1].text}"
    if NO_IMG:
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                },
            ]
        }
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": images[0],
            },
            {"type": "text", "text": query},
        ],
    }


def get_data(image_dir: str,dataset_path: str, dataset_name: str = "image_chat") -> List[dict[str, any]]:
    if dataset_name == "mmdd":
        test_data = MMDDData(
            path=dataset_path,
            to_filter=True,
            to_replace=True,
            image_path_by_url=create_image_path_by_url_mmdd(
                image_dir
            ),
            to_unroll=True,
            min_images_per_dialog=1,
            # n_samples=1100,
            to_split=True,
        )
    elif dataset_name == "image_chat":
        test_data = ImageChatData(
            path=dataset_path,
            to_filter=True,
            to_replace=True,
            image_path_by_url=create_image_path_by_url_image_chat(image_dir),
            to_unroll=True,
            min_images_per_dialog=1,
            n_samples=4800,
            to_split=True,
        )
    data = []
    for dialog, suffix in test_data:
        if len([i for i in dialog.utterances[:-1] if len(i.images) > 0]) == 0:
            continue
        data.append((dialog.idx, transform_dialog_data_to_message(dialog, suffix)))
    if SPLIT is not None:
        n_splits = 8
        split_indices = [int(i * (len(data) / n_splits)) for i in range(n_splits + 1)]
        start_idx = split_indices[SPLIT]
        end_idx = split_indices[SPLIT + 1] if SPLIT < n_splits - 1 else len(data)
        data = data[start_idx:end_idx]
    return data


def infer_batch(engine: "InferEngine", infer_requests: List["InferRequest"]):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer(infer_requests, request_config)
    return resp_list


def make_outer_batches(data: List[dict[str, any]]) -> List[List[dict[str, any]]]:
    outer_batches = []
    for i in range(0, len(data), OBS):
        batch = data[i : i + OBS]
        outer_batches.append(batch)
    return outer_batches


def process_outer_batch(
    outer_batch: List[dict[str, any]], engine: "InferEngine"
) -> None:
    infer_requests = [InferRequest(messages=[message]) for _, message in outer_batch]
    infer_idxs = [idx for idx, _ in outer_batch]
    resp = infer_batch(engine, infer_requests)
    pd.DataFrame(
        {
            "id": infer_idxs,
            "pred": [r.choices[0].message.content for r in resp],
        }
    ).to_csv(
        os.path.join(OUTPUT_DIR, OUTPUT_FILE),
        index=False,
        sep=",",
        mode="a",
        header=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument("--batch_size", type=int, default=BS,
                        help="inner batch size passed to PtEngine")
    parser.add_argument("--outer_batch_size", type=int, default=OBS,
                        help="how many items per outer batch")
    parser.add_argument("--model", default=None,
                        help="HF model name (leave blank to use hard-coded one)")
    parser.add_argument("--adapter", default=None,
                        help="LoRA / adapter checkpoint (leave blank to use hard-coded one)")
    parser.add_argument("--output_file", default=None,
                        help="filename to write preds to")
    parser.add_argument("--dataset", default="image_chat",
                        choices=["image_chat", "mmdd"],
                        help="dataset to use")
    parser.add_argument("--dataset_path", default="data/ImageChat/image_chat/test.csv",
                        help="dataset to use")
    parser.add_argument("--image_dir", default="../data/ImageChat/yfcc_images",
                        help="image directory")
    parser.add_argument("--split", type=int, default=0,
                        help="split number")
    args = parser.parse_args()
    
    
    BS = args.batch_size
    OBS = args.outer_batch_size
    if args.model:        MODEL = args.model
    if args.adapter:      ADAPTER = args.adapter
    if args.split is not None:  SPLIT = args.split
    if args.output_file:
        if SPLIT is not None:
            OUTPUT_FILE = f"{args.output_file}_split_{SPLIT}"
        else:
            OUTPUT_FILE = args.output_file
    
    model = MODEL
    if model == "google/paligemma2-3b-pt-224":
        torch._dynamo.config.disable = True 
    adapter = ADAPTER
    engine = PtEngine(model, max_batch_size=BS, adapters=[adapter],device_map="auto")
    dataset = get_data(args.image_dir, args.dataset_path, args.dataset)
    if args.resume:
        # Read existing file to get the last idx
        print(f"reading {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
        df = pd.read_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILE), sep=",")
        print(f"read {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
        already_present = set(df["id"].values)
        dataset = [d for d in dataset if d[0] not in already_present]
        print(f"skipping {len(df)} samples")
    # Create outer batches
    outer_batches = make_outer_batches(dataset)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not args.resume:
        # Write header
        pd.DataFrame(columns=["id", "pred"]).to_csv(
            os.path.join(OUTPUT_DIR, OUTPUT_FILE), index=False, sep=","
        )
    for outer_batch in tqdm(outer_batches):
        process_outer_batch(outer_batch, engine)
