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



MASTER_DATASET_PATH: str = "master_dataset.csv"
import numpy as np
np.random.seed(42)


MODEL_LIB_BY_TYPE = {
    # Model.MPC: 'router.models.mpc_model.mpc_model.MPCModel',
    Model.GENERAL: 'router.models.general.general_features_model.GeneralFeaturesModel',
    Model.QB: 'router.models.qb_model.queryblazer_model.QueryBlazerModel',
    # Model.MINICPM: 'router.models.minicpm.minicpm_model.MiniCPMModel',
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
    parser = argparse.ArgumentParser(description="Create a master dataset from MMDD data using a specified model.")
    # flag
    parser.add_argument('--run_models', action='store_true', help="Run models to create master dataset.")
    parser.add_argument('--combine_outputs', action='store_true', help="Combine outputs from all models into a single master dataset.")
    args = parser.parse_args()
    dataset = get_dataset(DatasetType.MMDD)
    paths = []

    master_dataset = pd.read_csv(MASTER_DATASET_PATH)


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
            paths.append(run_save_model(model_cls(), dataset, master_dataset))
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
        data = pd.read_csv("master_dataset.csv")
        print(f"Length of original master dataset: {len(data)}")
        df = pd.merge(data[['idx', 'pred', 'model']], df, on='idx', how='left')
        print(f"Length of combined master dataset: {len(df)}")
        df.to_csv("master_dataset_combined.csv", index=False)
        print("Combined master dataset saved to master_dataset_combined.csv")

    

    



