import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import logging
import joblib
import argparse

from router.utils.dataset_utils import DatasetType, get_combined_score, get_dataset, partial_f1
from router.utils.modelling_utils import load_combined_pred_by_idx # Moved import to the top

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
np.random.seed(42)





def train_random_forest_model(data_path: str, target_column: str, dataset_type: DatasetType):
    """
    Train a Random Forest model on the dataset located at data_path.
    Args:
        data_path (str): Path to the dataset CSV file.
        target_column (str): The column name in the dataset to predict.
        dataset_type (DatasetType): The type of dataset being used.
    Returns:
        model (RandomForestClassifier): The trained Random Forest model.
        label_encoders (dict): A dictionary of label encoders for categorical variables.
    """
    # Load the dataset
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} does not exist.")
        return None, None # Return two Nones as the original call expects two values
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return None, None

    df.set_index('idx', inplace=True, drop=False)  # Ensure 'idx' is set as index if it exists
    # Drop unnecessary columns if they exist (robustly)
    columns_to_drop = ['idx', 'suffix', 'nll', 'pred']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)
        logger.info(f"Dropped specified columns: {existing_columns_to_drop}")
    else:
        logger.info(f"No specified columns ({', '.join(columns_to_drop)}) found to drop.")

    # Drop rows with missing values
    initial_rows = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing values. Shape after dropna: {df.shape}")
    else:
        logger.info(f"No rows with missing values found. Shape remains: {df.shape}")

    if df.empty:
        logger.error("Dataset is empty after preprocessing. Cannot proceed.")
        return None, None
        
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataset after preprocessing.")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return None, None

    logger.info(f"Target column '{target_column}' value counts:\n{df[target_column].value_counts().to_string()}")

    # Encode categorical variables
    label_encoders = {}
    # Create a copy for encoding to avoid SettingWithCopyWarning if df is a slice
    df_encoded = df.copy() 
    for column in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        # Fit_transform on non-missing values, already handled by dropna, but good practice
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
        logger.info(f"Encoded categorical column: {column}")
    
    # Split the dataset into features and target variable
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    
    logger.info(f"Features are: {X.columns.tolist()}")
    logger.info(f"Target variable is '{target_column}'")
    if label_encoders:
        logger.info(f"There are {len(label_encoders)} categorical variables encoded: {list(label_encoders.keys())}")
    else:
        logger.info("No categorical variables were encoded.")
    logger.info(f"Target unique values (post-encoding if applicable): {y.nunique()}: {y.unique()}")

    if X.empty:
        logger.error("Feature set X is empty. This might happen if the target column was the only column or all other columns were dropped.")
        return None, None


    # set seed = 42

    # Split the data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Create and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
    
    # --- Feature Importance Analysis ---
    importances = model.feature_importances_
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    logger.info("Feature Importances:")
    # Log the full feature importance dataframe as a string
    logger.info("\n" + feature_importance_df.to_string())
    
    print("\nFeature Importances:")
    print(feature_importance_df)
    # --- End of Feature Importance Analysis ---

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0) # Added zero_division for robustness
    
    logger.info(f"Model Accuracy: {accuracy:.4f}") # Increased precision for accuracy
    logger.info(f"Classification Report:\n{report}")

    print(f"\nModel Accuracy: {accuracy:.4f}")
    # print which label means what using label encoders
    le = label_encoders.get(target_column, None)
    model_label_map = dict(zip(le.classes_, range(len(le.classes_))))
    rev_model_label_map = {v: k for k, v in model_label_map.items()}  # Reverse mapping
    print(f"label mapping: {model_label_map}")

    dataset = get_dataset(dataset_type)
    combined_pred_by_idx_filename = f"combined_pred_by_idx_{dataset_type.value}.json"
    combined_pred_by_idx = load_combined_pred_by_idx(combined_pred_by_idx_filename)
    y_pred_df = pd.DataFrame(list(y_pred), columns=[target_column])
    y_test_df = pd.DataFrame(y_test, columns=[target_column])
    y_pred_df['idx'] = X_test.index
    y_test_df['idx'] = X_test.index
    y_pred_df[target_column] = y_pred_df[target_column].map(lambda x: rev_model_label_map.get(x, x))  # Map predictions back to original labels
    print(y_pred_df[target_column].value_counts())
    y_test_df[target_column] = y_test_df[target_column].map(lambda x: rev_model_label_map.get(x, x))  # Map test labels back to original labels
    for model_name, label in model_label_map.items():
        print(f"If only {model_name} is used")
        temp_y_pred_df = y_pred_df.copy()
        temp_y_pred_df[target_column] = model_name
        score = get_combined_score(dataset, combined_pred_by_idx, temp_y_pred_df)
        print(f"Score for {model_name}: {score}")
    score = get_combined_score(dataset, combined_pred_by_idx, y_pred_df)
    print(f"Score using router: {score}")
    score = get_combined_score(dataset, combined_pred_by_idx, y_test_df)
    print(f"Max Score on test set: {score}")
    return model, label_encoders

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='master_dataset_combined.csv')
    parser.add_argument('--dataset', type=str, default='mmdd', choices=['mmdd', 'image_chat'])
    args = parser.parse_args()
    data_path = args.data_path
    target_column = 'model'
    dataset = args.dataset
    if dataset == "mmdd":
        dataset = DatasetType.MMDD
    elif dataset == "image_chat":
        dataset = DatasetType.IMAGECHAT
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    trained_model, encoders = train_random_forest_model(data_path, target_column, dataset)
    
    if trained_model and encoders:
        # Save the model and label encoders if needed
        joblib.dump(trained_model, 'random_forest_model.pkl')
        logger.info("Model saved to random_forest_model.pkl")
        joblib.dump(encoders, 'label_encoders.pkl')
        logger.info("Label encoders saved to label_encoders.pkl")
        print("\nModel and label encoders saved successfully.")
    else:
        logger.error("Model training failed or returned None. Model and encoders not saved.")
        print("\nModel training failed. Check logs for details. Model and encoders not saved.")
