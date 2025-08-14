from __future__ import annotations
import os
from re import T
from typing import Any, Dict, List, Tuple
from abc import abstractmethod
import ast
import pandas as pd
from tqdm import tqdm

import json
from datasets import Dataset
import pandas as pd
import numpy as np

import hashlib
import pickle
import logging

logger = logging.getLogger(__name__)


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


dup_filter = DuplicateFilter()

logger.addFilter(dup_filter)


TEST_CONFIG: Dict[str, Any] = {
    "to_filter": True,
    "to_replace": False,
    "to_unroll": True,
    "min_images_per_dialog": 1,
    "n_samples": None,
    "to_split": True,
}


def hash_dataset(dataset):
    dataset_bytes = pickle.dumps(dataset)
    hash_obj = hashlib.sha256()
    hash_obj.update(dataset_bytes)
    return hash_obj.hexdigest()


class Utterance:
    text: str
    images: List[str]
    captions: List[str] | None
    speaker: str | None

    def __init__(
        self,
        text: str,
        speaker: str | None = None,
        images: List[str] | None = None,
        captions: List[str] | None = None,
    ):
        self.text = text
        self.speaker = speaker
        self.images = images if images else []
        self.captions = captions
        if self.captions is not None:
            assert len(self.images) == len(
                self.captions
            ), f"Number of images ({len(self.images)}) and captions ({len(self.captions)}) should be equal"

    def split(self) -> List[Tuple[Utterance, str]]:
        l = len(self.text)
        splits = []
        for i in range(0, l + 1):
            utr = Utterance(self.text[:i], self.speaker, self.images, self.captions)
            splits.append((utr, self.text[i:]))
        return splits

    def add_image(self, image: str, caption: str | None = None):
        self.images.append(image)
        if caption is not None:
            if self.captions is None:
                self.captions = []
            self.captions.append(caption)
            assert len(self.images) == len(
                self.captions
            ), f"Number of images ({len(self.images)}) and captions ({len(self.captions)}) should be equal"

    def format_utterance(self, keep_image_path=False) -> str:
        if self.captions is not None and not keep_image_path:
            return "".join([f"[{caption}]" for caption in self.captions]) + self.text
        return "".join([f"[{image}]" for image in self.images] + [self.text])

    def __repr__(self) -> str:
        return self.format_utterance()


class Dialog:
    idx: str
    utterances: List[Utterance]

    def __init__(self, utterances: Any, idx: str = ""):
        if isinstance(utterances, list) and all(
            [isinstance(utterance, Utterance) for utterance in utterances]
        ):
            self.utterances = utterances
        else:
            self.utterances = self.get_utterances(utterances)

        self.idx = idx

    def __getitem__(self, item: int) -> Utterance:
        return self.utterances[item]

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self):
        return iter(self.utterances)

    def format_dialog(self, keep_image_path=False) -> str:
        return "\n".join(
            [self.idx]
            + [
                f"{utterance.speaker}: {utterance.format_utterance(keep_image_path)}"
                for utterance in self.utterances
            ]
        )

    def unroll(self) -> List[Dialog]:
        return [
            Dialog(self.utterances[:i], self.idx + "__u" + str(i))
            for i in range(1, len(self.utterances) + 1)
        ]

    def create_splits(self) -> List[Tuple[Dialog, str]]:
        context = self.utterances[:-1]
        response = self.utterances[-1]
        response_splits = response.split()
        return [
            (
                Dialog(context + [response_split[0]], self.idx + "__s" + str(i)),
                response_split[1],
            )
            for i, response_split in enumerate(response_splits)
        ]

    @property
    def context(self) -> List[Utterance]:
        return self.utterances[:-1]

    @property
    def response(self) -> Utterance:
        return self.utterances[-1]

    @property
    def character_count(self) -> int:
        """
        Sum of the character count in text of all utterances in the dialog. Images are not counted.
        """
        return sum([len(utterance.text) for utterance in self.utterances])

    @property
    def level(self) -> int:
        """
        Level of the dialog. Level is defined as following:
        Level 0: raw dialogs with no unrolling or splitting. have idx of the form "te_bst:806"
        Level 1: unrolled dialogs. have idx of the form "te_bst:806__u11"
        Level 2: split dialogs. have idx of the form "te_bst:806__u11__s0"
        """
        unrolled_flag = "__u" in self.idx
        split_flag = "__s" in self.idx
        if split_flag:
            return 2
        if unrolled_flag:
            return 1
        return 0

    @abstractmethod
    def get_utterances(self, inp: Any) -> List[Utterance]:
        pass


class DialogCC(Dialog):
    def get_utterances(self, inp) -> List[Utterance]:
        assert isinstance(inp, str), "Input should be a string for DialogCC"
        utterances = []
        ls = ast.literal_eval(inp)
        for dictionary in ls:
            utterance = Utterance(
                text=dictionary["utterance"], speaker=dictionary["speaker"]
            )
            for image in dictionary["shared_image"]:
                utterance.add_image(image["image_url"], image["caption"])
            utterances.append(utterance)
        return utterances

    def __repr__(self) -> str:
        return self.format_dialog()


class PhotoChat(Dialog):
    def get_utterances(self, inp) -> List[Utterance]:
        assert isinstance(
            inp, pd.Series
        ), "Input should be a pandas Series for PhotoChat"
        utterances = []
        # print(f"{type(inp[0])=}  {inp=}")
        total_dict = ast.literal_eval(str(inp["dialogue"]))
        ls = total_dict["dialogue"]
        assert (
            total_dict["photo_id"].replace("/", "_") == inp["image_id"]
        ), f"Error in Dataset, {total_dict['photo_id']} != {inp['image_id']}"
        # print(f"{(ls['dialogue'])=}")
        image_filled_flag = False
        for dictionary in ls:
            utterance = Utterance(
                text=dictionary["message"], speaker=dictionary["user_id"]
            )
            if dictionary["share_photo"] is True:
                if not image_filled_flag:
                    # print(f"{inp['image_id']=}  {dictionary['photo_description']=}")
                    utterance.add_image(inp["image_id"], total_dict["photo_description"])
                    image_filled_flag = True
                else:
                    raise ValueError(
                        "Multiple images found in a single dialog in PhotoChat"
                    )
            utterances.append(utterance)
        return utterances

class ImageChat(Dialog):
    def get_utterances(self, inp: str) -> List[Utterance]:
        utterances = []
        ls = ast.literal_eval(inp)
        for dictionary in ls: 
            utterance = Utterance(
                text=dictionary["utterance"], speaker=dictionary["speaker"]
            )
            if dictionary["image_hash"] != "":
                utterance.add_image(dictionary["image_hash"])
            utterances.append(utterance)

        return utterances

    def __repr__(self) -> str:
        return self.format_dialog()
    
class MMDD(Dialog):
    def get_utterances(self, inp) -> List[Utterance]:
        assert isinstance(
            inp, pd.Series
        ), "Input should be a pandas Series for MMDD"
        utterances = []
        # print(f"{type(inp[0])=}  {inp=}")
        total_dict = json.loads(inp["dialogue"])
        ls = total_dict["dialogue"]
        # print(f"{(ls['dialogue'])=}")
        image_filled_flag = False
        for dictionary in ls:
            utterance = Utterance(
                text=dictionary["message"], speaker=dictionary["user_id"]
            )
            if dictionary["share_photo"] is True:
                if not image_filled_flag:
                    # print(f"{inp['image_id']=}  {dictionary['photo_description']=}")
                    utterance.add_image(inp["image_id"], dictionary["message"])
                    image_filled_flag = True
                else:
                    raise ValueError(
                        "Multiple images found in a single dialog in MMDD"
                    )
            utterances.append(utterance)
        return utterances

class DialogData(Dataset):
    def __init__(
        self,
        path: str | Tuple[List[Dialog], List[str]],
        to_preprocess: bool = False,
        to_filter: bool = False,
        to_replace: bool = False,
        image_path_by_url: Dict[str, str] = {},
        to_unroll: bool = False,
        min_images_per_dialog: int | None = None,
        n_samples: int | None = None,
        to_split: bool = False,
    ):
        """
        Custom Dataset class extending Hugging Face's Dataset.
            **NOTE**: Preprocessing is done in the constructor only when the dataset is loaded from a file. i.e. when path is a string.
        Args:
            path (str or List of Tuple of List of Dialogs and List of Suffixes): Path to the dataset file or Tuple of List of Dialogs and List of Suffixes.
            to_preprocess (bool): If True, preprocesses the dialogs.
            to_filter (bool): If True, filters out dialogs with images that are not in the image_path_by_url dictionary or whose image paths do not exist.
            to_replace (bool): If True, replaces image urls with image paths.
            image_path_by_url (Dict[str, str]): Dictionary mapping image urls to image paths.
            to_unroll (bool): If True, unrolls dialogs into multiple dialogs.
            min_images_per_dialog (int): Minimum number of images per dialog.
            n_samples (int): Number of samples to sample from EACH CATEGORY of the dataset.
            to_split (bool): If True, splits dialogs into multiple dialogs.
        Returns:
            None
        """
        if not isinstance(path, str):
            # assertion that all other flags are off if path is not a string
            args = locals()
            args.pop("self")
            args.pop("path")
            assert all(
                [not value for value in args.values()]
            ), "All other flags/args should be off/empty if path is not a string"
        self.to_preprocess = to_preprocess
        self.to_unroll = to_unroll
        self.to_split = to_split
        self.to_filter = to_filter
        self.to_replace = to_replace
        self.n_samples = n_samples
        self.min_images_per_dialog = min_images_per_dialog
        self.path = path
        self.dialogs = self.parse_raw_file() if isinstance(path, str) else path[0]
        self.suffixes = [] if isinstance(path, str) else path[1]
        if to_preprocess:
            self.dialogs = [self.preprocess(dialog) for dialog in self.dialogs]
        self.n_samples = n_samples
        self.image_path_by_url = image_path_by_url
        if to_filter:
            assert (
                len(image_path_by_url) > 0
            ), "image_path_by_url dictionary must be provided if to_filter is True"
            self.filter_and_replace_image_paths()
        if to_unroll:
            unrolled_dialogs = []
            for dialog in tqdm(self.dialogs, desc="Unrolling dialogs"):
                unrolled_dialogs += dialog.unroll()
            self.dialogs = unrolled_dialogs
        if min_images_per_dialog is not None:
            self.dialogs = [
                dialog
                for dialog in self.dialogs
                if sum([len(utterance.images) for utterance in dialog])
                >= min_images_per_dialog
            ]
        if n_samples is not None:
            self.dialogs = self.sample(n_samples)
        if to_split:
            split_dialogs = []
            suffixes = []
            for dialog in tqdm(self.dialogs, desc="Splitting dialogs"):
                for split_dialog, suffix in dialog.create_splits():
                    suffixes.append(suffix)
                    split_dialogs.append(split_dialog)
            self.dialogs = split_dialogs
            self.suffixes = suffixes
        else:
            self.suffixes = [None] * len(self.dialogs)
        # duplicate fix 
        freq_by_idx = {}
        self.dialog_suffix_by_id = {}
        for dialog, suffix in zip(self.dialogs, self.suffixes):
            if dialog.idx not in freq_by_idx:
                freq_by_idx[dialog.idx] = 0
            else:
                freq_by_idx[dialog.idx] += 1
            prefix = dialog.idx.split("_", 1)[0]
            actual_idx = dialog.idx.split("_", 1)[1]
            dialog.idx = f"{prefix}_##{freq_by_idx[dialog.idx]}_{actual_idx}"
            self.dialog_suffix_by_id[dialog.idx] = (dialog, suffix)
        print(
            f"Total dialogs: {len(self)}, Total suffixes: {len(self.suffixes)}, to_filter: {to_filter}, to_replace: {to_replace}, image_path_by_url (size): {len(image_path_by_url)}, to_unroll: {to_unroll}, min_images_per_dialog: {min_images_per_dialog}, n_samples: {n_samples}, to_split: {to_split}"
        )
        print(
            "HASH: ",
            hash_dataset(
                [[utterance.text for utterance in dialog] for dialog in self.dialogs]
            ),
        )

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx: int) -> Tuple[Dialog, str | None]:
        if self.to_split:
            return self.dialogs[idx], self.suffixes[idx]
        return self.dialogs[idx], None

    def __iter__(self):
        return iter(zip(self.dialogs, self.suffixes))

    def filter_and_replace_image_paths(self):
        """
        Filters out dialogs with images that are not in the image_path_by_url dictionary or whose image paths do not exist.
        """
        dialogs = []
        print("Reading image files")
        for dialog in tqdm(self.dialogs, desc="Filtering dialogs"):
            not_found_flag = False
            utterances = []
            for utterance in dialog.utterances:
                images_renamed = [
                    self.image_path_by_url[image]
                    for image in utterance.images
                    if image in self.image_path_by_url
                    and os.path.exists(self.image_path_by_url[image])
                ]
                if len(images_renamed) != len(utterance.images):
                    not_found_flag = True
                    break
                utterances.append(
                    Utterance(
                        utterance.text,
                        utterance.speaker,
                        (images_renamed if self.to_replace else utterance.images),
                        utterance.captions,
                    )
                )
            if not not_found_flag:
                dialogs.append(Dialog(utterances, dialog.idx))
        self.dialogs = dialogs

    @classmethod
    def preprocess(cls, dialog: Dialog) -> Dialog:
        """
        Preprocesses the dialog.
        """
        for utterance in dialog.utterances:
            utterance.text = utterance.text.lower()
            if (
                dialog.level != 2
            ):  # since when level is 2 the trailing spaces actually matter
                utterance.text = utterance.text.strip()
            else:
                # log once across all dialogs
                logger.warning(f"Dialog level 2 found: Hence not stripping")
            if utterance.speaker:
                utterance.speaker = utterance.speaker.lower().strip()
        return dialog

    @abstractmethod
    def parse_raw_file(self) -> List[Dialog]:
        """
        Shall parse the dataset file and return a list of Dialog objects.
        """
        pass

    @abstractmethod
    def sample(self, n: int) -> List[Dialog]:
        """
        Shall sample n dialogs from each category of the dataset.
        """
        raise NotImplementedError("Sampling not yet implemented")


class DialogCCData(DialogData):
    def id_prefix(self, path: str) -> str:
        """
        Returns the prefix of the dialog id.
        Example:
            path = "data/DialogCC/test.csv"
            id_prefix(path) = "te"
        """

        return path.split("/")[-1].split(".")[0][:2]

    def parse_raw_file(self) -> List[Dialog]:
        """
        Parses the raw dataset file and returns a list of Dialog objects.
        """
        assert isinstance(self.path, str), "Path must be a string"
        df = pd.read_csv(self.path)
        dialogs = []
        for idx, row in tqdm(df.iterrows()):
            dialog = DialogCC(
                row["dialogue"],
                f"{self.id_prefix(self.path)}_{row['dialogue_id']}",
            )
            dialogs.append(dialog)
        return dialogs

    def sample(self, n: int) -> List[Dialog]:
        """
        Samples n dialogs from each category of the dataset.
        """
        dialogs_by_category = {}
        for dialog in self.dialogs:
            category = dialog.idx.split(":")[0]
            if category not in dialogs_by_category:
                dialogs_by_category[category] = []
            dialogs_by_category[category].append(dialog)
        print("Number of dialogs found in each category:")
        for category in dialogs_by_category:
            print(f"{category}: {len(dialogs_by_category[category])}")
        sampled_dialogs = []
        for category in dialogs_by_category:
            rng = np.random.default_rng(seed=42)
            total = len(dialogs_by_category[category])
            if n > total:
                raise ValueError(
                    f"Number of samples to sample from category {category} is greater than the total number of dialogs in the category."
                )
            else:
                # choose n random numbers from 0 to total-1
                sample_indices = rng.choice(total, n, replace=False)
                sampled_dialogs += [
                    dialogs_by_category[category][i] for i in sample_indices
                ]
        return sampled_dialogs


class PhotoChatData(DialogData):
    def id_prefix(self, path: str) -> str:
        """
        Returns the prefix of the dialog id.
        """
        return path.split("/")[-1].split(".")[0][:2]

    def parse_raw_file(self) -> List[Dialog]:
        """
        Parses the raw dataset file and returns a list of Dialog objects.
        """
        assert isinstance(self.path, str), "Path must be a string"
        df = pd.read_csv(self.path)
        dialogs = []
        for idx, row in tqdm(df.iterrows()):
            dialog = PhotoChat(row, f"{self.id_prefix(self.path)}_{row['dialogue_id']}")
            dialogs.append(dialog)
        return dialogs

    def sample(self, n: int) -> List[Dialog]:
        """
        Randomly sample n dialogs from the dataset.
        """
        rng = np.random.default_rng(seed=42)
        total = len(self.dialogs)
        if n > total:
            raise ValueError(
                f"Number of samples to sample is greater than the total number of dialogs."
            )
        else:
            # choose n random numbers from 0 to total-1
            sample_indices = rng.choice(total, n, replace=False)
            sampled_dialogs = [self.dialogs[i] for i in sample_indices]
        return sampled_dialogs
    
class ImageChatData(DialogData):
    def id_prefix(self, path: str) -> str:
        """
        Returns the prefix of the dialog id.
        Example:
            path = "data/DialogCC/test.csv"
            id_prefix(path) = "te"
        """

        return path.split("/")[-1].split(".")[0][:2]

    def parse_raw_file(self) -> List[Dialog]:
        """
        Parses the raw dataset file and returns a list of Dialog objects.
        """
        assert isinstance(self.path, str), "Path must be a string"
        df = pd.read_csv(self.path)
        dialogs = []
        for idx, row in tqdm(df.iterrows()):
            dialog = ImageChat(
                row["dialogue"],
                row["id"],
            )
            dialogs.append(dialog)
        return dialogs

    def sample(self, n: int) -> List[Dialog]:
        """
        Samples n dialogs from the dataset.
        """
        rng = np.random.default_rng(seed=42)
        total = len(self.dialogs)
        if n > total:
            raise ValueError(
                f"Number of samples to sample is greater than the total number of dialogs."
            )
        else:
            # choose n random numbers from 0 to total-1
            sample_indices = rng.choice(total, n, replace=False)
            return [self.dialogs[i] for i in sample_indices]
    
class MMDDData(DialogData):
    def id_prefix(self, path: str) -> str:
        """
        Returns the prefix of the dialog id.
        """
        return path.split("/")[-1].split(".")[0][:2]
    
    def parse_raw_file(self) -> List[Dialog]:
        """
        Parses the raw dataset file and returns a list of Dialog objects.
        """
        assert isinstance(self.path, str), "Path must be a string"
        df = pd.read_csv(self.path)
        dialogs = []
        for idx, row in tqdm(df.iterrows()):
            dialog = MMDD(row, f"{self.id_prefix(self.path)}_{row['dialogue_id']}")
            dialogs.append(dialog)
        return dialogs

    def sample(self, n: int) -> List[Dialog]:
        """
        Randomly sample n dialogs from the dataset.
        """
        rng = np.random.default_rng(seed=42)
        total = len(self.dialogs)
        if n > total:
            raise ValueError(
                f"Number of samples to sample is greater than the total number of dialogs."
            )
        else:
            # choose n random numbers from 0 to total-1
            sample_indices = rng.choice(total, n, replace=False)
            sampled_dialogs = [self.dialogs[i] for i in sample_indices]
        return sampled_dialogs



def create_image_path_by_url(image_names_dir: str, images_dir: str) -> Dict[str, str]:
    image_path_by_url = {}
    for file in os.listdir(image_names_dir):
        with open(os.path.join(image_names_dir, file)) as f:
            image_names = pd.read_csv(f, sep="\t", header=None)
            for i in range(len(image_names)):
                image_path_by_url[image_names.iloc[i, 0]] = os.path.join(
                    images_dir, str(image_names.iloc[i, 1])
                )
    return image_path_by_url


def create_image_path_by_url_photochat(images_dir: str) -> Dict[str, str]:
    image_path_by_url = {}
    for f in filter(lambda x: x.endswith(".jpg"), os.listdir(images_dir)):
        image_path_by_url[f.split(".")[0]] = os.path.join(images_dir, f)
    return image_path_by_url

def create_image_path_by_url_image_chat(
    images_dir: str
) -> Dict[str, str]:
    image_path_by_url = {}
    for file in os.listdir(images_dir):
        # filename would be of form <image_hash>.jpg
        image_hash = file.split(".")[0]
        image_path_by_url[image_hash] = os.path.join(images_dir, file)
    return image_path_by_url

def create_image_path_by_url_mmdd(images_dir: str) -> Dict[str, str]:
    image_path_by_url = {}
    
    for root, _, files in os.walk(images_dir):  # Recursively iterate through subdirectories
        for f in files:
            if f.endswith(".jpg"):  # Process only JPG files
                image_path_by_url[f] = os.path.join(root, f)
                # print(f"Image path by url: {f} -> {image_path_by_url[f]}")
    return image_path_by_url


def intersection(
    data1: DialogData, data2: DialogData
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns the number of repeated responses in data1 and data2.
    Args:
        data1 (DialogData): First dataset.
        data2 (DialogData): Second dataset.
    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: Number of repeated responses in data2 of data1 and data1 of data2.

    Note:
        data1 and data2 should not be split. This only checks the last utterance of each dialog.
    """
    assert (
        data1.to_split == False and data2.to_split == False
    ), "Data should not be split"

    freq1 = {}
    freq2 = {}
    for dialog, _ in tqdm(data1, desc="Processing data1"):
        freq1[dialog.response.text] = freq1.get(dialog.response.text, 0) + 1
    for dialog, _ in tqdm(data2, desc="Processing data2"):
        freq2[dialog.response.text] = freq2.get(dialog.response.text, 0) + 1

    repeated1 = {}
    repeated2 = {}
    for dialog, _ in tqdm(data1, desc="Checking repeated in data1"):
        if freq2.get(dialog.response.text, 0) > 0:
            repeated1[dialog.idx] = freq2[dialog.response.text]
    for dialog, _ in tqdm(data2, desc="Checking repeated in data2"):
        if freq1.get(dialog.response.text, 0) > 0:
            repeated2[dialog.idx] = freq1[dialog.response.text]
    print(f"Repeated in data1: {len(repeated1)}, Repeated in data2: {len(repeated2)}")
    return (repeated1, repeated2)


def update_intersection():
    test_data = ImageChatData(
        path="/home/anubhab-pg/anubhab/ParlAI/data/image_chat/test.csv",
        image_path_by_url=create_image_path_by_url_image_chat("/home/anubhab-pg/anubhab/ParlAI/data/yfcc_images"),
        to_filter=True,
        to_replace=True,
        to_unroll=True,
        min_images_per_dialog=1,
        n_samples=None,
        to_split=False,
    )
    train_data = ImageChatData(
        path="/home/anubhab-pg/anubhab/ParlAI/data/image_chat/train.csv",
        image_path_by_url=create_image_path_by_url_image_chat("/home/anubhab-pg/anubhab/ParlAI/data/yfcc_images"),
        to_filter=True,
        to_replace=True,
        to_unroll=True,
        min_images_per_dialog=1,
        n_samples=None,
        to_split=False,
    )
    print("Calculating Intersection:")
    repeated_data = intersection(train_data, test_data)
    with open("data/ImageChat/intersection_san.json", "w") as f:
        json.dump({"train_key": repeated_data[0], "test_key": repeated_data[1]}, f)


def update_intersection_mmdd():
    test_data = MMDDData(
        path="data/MMDD/test.csv",
        image_path_by_url=create_image_path_by_url_mmdd("data/MMDD/images"),
        to_filter=True,
        to_replace=True,
        to_unroll=True,
        min_images_per_dialog=1,
        n_samples=None,
        to_split=False,
    )
    train_data = MMDDData(
        path="data/MMDD/train.csv",
        image_path_by_url=create_image_path_by_url_mmdd("data/MMDD/images"),
        to_filter=True,
        to_replace=True,
        to_unroll=True,
        min_images_per_dialog=1,
        n_samples=None,
        to_split=False,
    )
    print("Calculating Intersection:")
    repeated_data = intersection(train_data, test_data)
    with open("data/MMDD/intersection_san.json", "w") as f:
        json.dump({"train_key": repeated_data[0], "test_key": repeated_data[1]}, f)



def update_test_formatted(test_data: DialogData):
    """
    Updates the test data with the formatted dialog.
    makes a csv file with the following columns:
    dialogue_id,prefix,suffix
    """
    ids = []
    prefixes = []
    suffixes = []
    for dialog, suffix in tqdm(test_data, desc="Formatting dialogs"):
        ids.append(dialog.idx)
        prefixes.append(dialog.response.text.lower())
        suffixes.append(str(suffix).lower())
    
    with open("data/ImageChat/test_formatted.txt", "w") as f:
        for p,s in zip(prefixes, suffixes):
            f.write(f"{p}\t{s}\n")    
    
    df = pd.DataFrame({"id": ids, "prefix": prefixes, "suffix": suffixes})
    df.to_csv("data/ImageChat/test_formatted.csv", index=False)


def update_train_txt(train_data: DialogData, filename: str):
    sents = []
    for dialog, _ in train_data:
        sents.append(dialog.response.text.lower())
    with open(f"data/ImageChat/{filename}", "w") as f:
        f.write("\n".join(sents))


if __name__ == "__main__":
    pass
    # update_intersection()
    # update_intersection_mmdd()
    # train_data = ImageChatData(
    #         path="/home/anubhab-pg/anubhab/ParlAI/data/image_chat/train.csv",
    #         to_filter=True,
    #         to_replace=False,
    #         image_path_by_url=create_image_path_by_url_image_chat("/home/anubhab-pg/anubhab/ParlAI/data/yfcc_images"),
    #         to_unroll=True,
    #         min_images_per_dialog=1,
    #         # n_samples=1100,
    #         to_split=False,
    #     )
        
    # test_data = ImageChatData(
    #     path="/home/anubhab-pg/anubhab/ParlAI/data/image_chat/test.csv",
    #     to_filter=True,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url_image_chat("/home/anubhab-pg/anubhab/ParlAI/data/yfcc_images"),
    #     to_unroll=True,
    #     min_images_per_dialog=1,
    #     n_samples=4800,
    #     to_split=True,
    # )
    
    # print(test_data[100][0].format_dialog())

    # print(len(test_data))
    # # # exit(0)
    # update_test_formatted(test_data)
    # update_train_txt(train_data, 'train.txt')
    # test_data = MMDDData(
    #     path="data/MMDD/test.csv",
    #     to_filter=True,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url_mmdd("./data/MMDD/images"),
    #     to_unroll=True,
    #     min_images_per_dialog=1,
    #     # n_samples=1100,
    #     to_split=True,
    # )

    # print(test_data[10][0].format_dialog())

    # print(test_data.dialog_suffix_by_id["te_##7_persona_chat1680__u14__s43"][0].format_dialog())

    # update_train_txt(test_data, 'test.txt')
    
    # print(create_image_path_by_url_photochat("./data/PhotoChat/images"))
    # test_data = PhotoChatData(
    #     path="data/PhotoChat/train.csv",
    #     to_filter=True,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url_photochat("./data/PhotoChat/images"),
    #     # to_unroll=True,
    #     min_images_per_dialog=1,
    #     # n_samples=1100,
    #     to_split=False,
    # )
    # train_data = PhotoChatData(
    #     path="data/PhotoChat/train.csv",
    #     to_filter=True,
    #     to_replace=True,
    #     image_path_by_url=create_image_path_by_url_photochat("./data/PhotoChat/images"),
    #     # to_unroll=True,
    #     min_images_per_dialog=1,
    #     # n_samples=1100,
    #     to_split=False,
    # )
    # train_data = MMDDData(
    #     path="data/MMDD/train_new.csv",
    #     to_filter=False,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url_mmdd("./data/MMDD/images"),
    #     # to_unroll=True,
    #     min_images_per_dialog=1,
    #     # n_samples=1100,
    #     to_split=False,
    # )
    # train_data = DialogCCData(
    #     path="data/DialogCC/train.csv",
    #     to_filter=True,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url(
    #         "../tmp/image_names", "../tmp/images_n"
    #     ),
    #     to_unroll=False,
    #     min_images_per_dialog=1,
    #     n_samples=None,
    #     to_split=False,
    # )
    # test_data = MMDDData(
    #     path="data/MMDD/test.csv",
    #     to_filter=False,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url_mmdd("./data/MMDD/images"),
    #     # to_unroll=True,
    #     min_images_per_dialog=1,
    #     # n_samples=1100,
    #     to_split=False,
    # )
    # test_data = DialogCCData(
    #     path="data/DialogCC/test.csv",
    #     to_filter=True,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url(
    #         "../tmp/image_names", "../tmp/images_n"
    #     ),
    #     to_unroll=True,
    #     min_images_per_dialog=1,
    #     n_samples=1100,
    #     to_split=False,
    # )

    # print([dialog.idx for dialog, _ in train_data][:100])
    # while True:
    #     i = int(input())
    #     print(train_data[i][0].format_dialog())
    # print(test_data.dia)

    # print(len(test_data.dialogs))
    # print(len(test_data.dialog_suffix_by_id))
    # val_data = DialogCCData(
    #     path="data/DialogCC/validation.csv",
    #     to_filter=True,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url(
    #         "../tmp/image_names", "../tmp/images_n"
    #     ),
    #     to_unroll=True,
    #     min_images_per_dialog=1,
    #     n_samples=None,
    #     to_split=True,
    # )
    # train_data = DialogCCData(
    #     path="data/DialogCC/train.csv",
    #     to_filter=True,
    #     to_replace=False,
    #     image_path_by_url=create_image_path_by_url(
    #         "../tmp/image_names", "../tmp/images_n"
    #     ),
    #     to_unroll=True,
    #     min_images_per_dialog=1,
    #     n_samples=None,
    #     to_split=True,
    # )
    # print(f"length of test_data = {len(test_data)}")
    # print(f"length of val_data = {len(val_data)}")
    # print(f"length of train_data = {len(train_data)}")
    # update_train_txt(train_data)
    # update_test_formatted(dialog_data)
    # print(dialog_data.dialog_suffix_by_id["te_bst:806__u11"][0].format_dialog())
    # print(len(dialog_data))
    # print(
    #     "avg number of images = ",
    #     sum(
    #         [
    #             sum([len(utterance.images) for utterance in dialog])
    #             for dialog, _ in test_data
    #         ]
    #     )
    #     / len(test_data),
    # )
    # print(
    #     "sum of images = ",
    #     sum(
    #         [
    #             sum([len(utterance.images) for utterance in dialog])
    #             for dialog, _ in test_data
    #         ]
    #     ),
    # )
    # # avg number of characters per dialog
    # print(
    #     "avg number of characters = ",
    #     sum([dialog.character_count for dialog, _ in test_data])
    #     / len(test_data),
    # )
    # print(test_data[0][0].format_dialog())
    # print(f"\nSuffix: {test_data[0][1]}")
