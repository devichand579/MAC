# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Literal

from tqdm import tqdm

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
OBS = 32         # outer-batch size
BS = 16          # inner batch size
OUTPUT_DIR = "output/"


NO_IMG = False


# OUTPUT_FILE = "out.all.imagechat.paligemma2.3b.pt224"
# torch._dynamo.config.disable = True      # for paligemma its required
# ADAPTER= "ckpts/exp_output_paligemma_imgchat/v3-20250529-040720/checkpoint-105000"
# MODEL = "google/paligemma2-3b-pt-224"

# OUTPUT_FILE = "out.all.imagechat.qwen2.vl.2b.instruct"
# ADAPTER= "ckpts/exp_output_qwen2_vl_imagechat/v1-20250529-055538/checkpoint-103000"
# MODEL = "qwen/Qwen2-VL-2B-Instruct"



# THRESHOLD = 0.8
# OBS = 1
# BS = 1
# OUTPUT_FILE = f"out.all.mmdd.minicpm.no_img"
# OUTPUT_DIR = "output/"
# NO_IMG = True
# ADAPTER= "/home/anubhab-pg/sm745052/swift/output/MiniCPM-V-2_6/v5-20250320-065856/checkpoint-2500"
# MODEL="openbmb/MiniCPM-V-2_6"


# OUTPUT_FILE = f"out.all.mmdd.minicpm"
# OUTPUT_DIR = "output/"
# NO_IMG = False
# ADAPTER= "output/MiniCPM-V-2_6/v27-20250320-103302/checkpoint-3000"
# MODEL="openbmb/MiniCPM-V-2_6"

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
    parser.add_argument("--dataset_path", default="../../anubhab/ParlAI/data/image_chat/test.csv",
                        help="dataset to use")
    parser.add_argument("--image_dir", default="../../anubhab/ParlAI/data/yfcc_images",
                        help="image directory")
    args = parser.parse_args()
    
    
    BS = args.batch_size
    OBS = args.outer_batch_size
    if args.model:        MODEL = args.model
    if args.adapter:      ADAPTER = args.adapter
    if args.output_file:  OUTPUT_FILE = args.output_file
    
    model = MODEL
    if model == "google/paligemma2-3b-pt-224":
        torch._dynamo.config.disable = True 
    adapter = ADAPTER
    engine = PtEngine(model, max_batch_size=BS, adapters=[adapter], device_map="auto")
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
