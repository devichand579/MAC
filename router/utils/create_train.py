import sys
sys.path.append(".")
from router.utils.modelling_utils import BaseModel
from router.utils.dataset_utils import get_dataset, DatasetType
from router.models.main import Model
from chatas.code.utils.dataset import Dialog, DialogData, MMDDData, create_image_path_by_url_mmdd
import importlib
import pandas as pd
from tqdm import tqdm
import argparse
import os



import numpy as np
np.random.seed(42)


MODEL_LIB_BY_TYPE = {
    # Model.MPC: 'router.models.mpc_model.mpc_model.MPCModel',
    # Model.GENERAL: 'router.models.general.general_features_model.GeneralFeaturesModel',
    Model.QB: 'router.models.qb_model.queryblazer_model.QueryBlazerModel',
    # Model.MINICPM: 'router.models.minicpm.minicpm_model.MiniCPMModel',
    # Model.PALIGEMMA: 'router.models.paligemma.paligemma_model.PaligemmaModel',
    # Model.QWEN: 'router.models.qwen.qwen_model.QwenModel',
}



def run_save_model(model: BaseModel, dataset: DialogData, master_dataset: pd.DataFrame) -> str:
    data = []
    for id in tqdm(master_dataset['idx'], desc=f"Processing {model.__class__.__name__}"):
        dialog, _ = dataset.dialog_suffix_by_id[id]
        # print(type(model))
        signals = model.get_signals(dialog)
        data.append({
            'idx': dialog.idx,
            **signals
        })
    df = pd.DataFrame(data)
    path = f"master_dataset_{model.__class__.__name__}.csv"
    df.to_csv(path, index=False)
    print(f"Master dataset saved to {path}")
    return path


def import_library(model_class: str):
    """
    Import a library dynamically based on the model class string.
    
    :param model_class: The full path of the model class to import.
    :return: The imported model class.
    """
    module_name, class_name = model_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a master dataset of features using a specified model.")
    # flag
    parser.add_argument('--run_models', action='store_true', help="Run models to create master dataset of features.")
    parser.add_argument('--master_dataset_path', type=str, default="master_dataset.csv", help="Path to the master dataset of features.")
    parser.add_argument('--combine_outputs', action='store_true', help="Combine outputs from all models into a single master dataset of features.")
    parser.add_argument('--dataset', type=str, default="mmdd", choices=["mmdd", "image_chat"], help="Dataset type to use (default: mmdd)")
    args = parser.parse_args()
    if args.dataset == "mmdd":
        dataset_type = DatasetType.MMDD
    elif args.dataset == "image_chat":
        dataset_type = DatasetType.IMAGECHAT
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    dataset = get_dataset(dataset_type)
    paths = []

    # Use dataset-specific master dataset filename if not provided
    if args.master_dataset_path == "master_dataset.csv":
        dataset_name = dataset_type.value
        args.master_dataset_path = f"master_dataset_{dataset_name}.csv"
    
    master_dataset = pd.read_csv(args.master_dataset_path)


    if args.run_models:
        # Run models to create master dataset
        for MODEL in Model.__members__.values():
            if MODEL not in MODEL_LIB_BY_TYPE:
                print(f"Model {MODEL} is not supported.")
                continue
            print(f"Running model: {MODEL.value}")
            model_class = MODEL_LIB_BY_TYPE[MODEL]
            model_cls = import_library(model_class)
            # print(f"Model library imported: {model_lib}")
            
            # Pass dataset-specific checkpoint directory for QB model
            if MODEL == Model.QB:
                if args.dataset == "mmdd":
                    ckpt_dir = 'QB_ckpts/QB_MMDD'
                elif args.dataset == "image_chat":
                    ckpt_dir = 'QB_ckpts/QB_ImageChat'
                else:
                    ckpt_dir = 'QB_ckpts/QB_MMDD'  # default
                model_instance = model_cls(ckpt_dir=ckpt_dir)
            else:
                model_instance = model_cls()
            
            paths.append(run_save_model(model_instance, dataset, master_dataset))
    else:
        paths = [path for path in os.listdir(".") if path.endswith('.csv') and "master_dataset_" in path and "combined" not in path]
    
    if args.combine_outputs:
        df = pd.DataFrame()
        for path in paths:
            print(f"Loading data from {path}")
            if df.empty:
                df = pd.read_csv(path)
            else:
            # If df is not empty, merge by 'idx' column
                data = pd.read_csv(path)
                df = pd.merge(df, data, on='idx', how='left')
        # Use dataset-specific master dataset filename
        dataset_name = dataset_type.value
        master_dataset_filename = f"master_dataset_{dataset_name}.csv"
        data = pd.read_csv(master_dataset_filename)
        print(f"Length of original master dataset: {len(data)}")
        df = pd.merge(data[['idx', 'pred', 'model']], df, on='idx', how='left')
        print(f"Length of combined master dataset: {len(df)}")
        combined_filename = f"master_dataset_combined_{dataset_name}.csv"
        df.to_csv(combined_filename, index=False)
        print(f"Combined master dataset saved to {combined_filename}")

    

    



