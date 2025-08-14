import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import logging
import joblib

# Import LightGBM specific classifier
from lightgbm import LGBMClassifier 

from router.utils.dataset_utils import DatasetType, get_combined_score, get_dataset
from router.utils.modelling_utils import load_combined_pred_by_idx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_lightgbm_model(data_path: str, target_column: str, dataset_type: DatasetType):
    """
    Train a LightGBM model on the dataset located at data_path.
    Args:
        data_path (str): Path to the dataset CSV file.
        target_column (str): The column name in the dataset to predict.
        dataset_type (DatasetType): The type of dataset being used.
    Returns:
        model (LGBMClassifier): The trained LightGBM model.
        preprocessing_pipeline (dict): A dictionary containing label encoders.
    """
    # Load the dataset
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} does not exist.")
        return None, None
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return None, None

    # Handle 'idx' column: set as index for potential lookup, then drop from features
    df.set_index('idx', inplace=True, drop=False) 

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
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
        logger.info(f"Encoded categorical column: {column}")
    
    # Split the dataset into features and target variable
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column] # Target column 'y' will be encoded by LabelEncoder for LightGBM

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

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Create and train the LightGBM model
    # LightGBM handles multi-class classification with 'multiclass' objective.
    # It also has native support for categorical features if you pass their names/indices
    # via the `categorical_feature` parameter. Since we've already LabelEncoded all 
    # 'object' columns to integers, LightGBM will treat them as numerical unless specified.
    # For simplicity and consistency with previous implementations, we'll let it treat them as numerical for now.
    model = LGBMClassifier(
        objective='multiclass', 
        num_class=y.nunique(), # Number of unique classes in target
        metric='multi_logloss', # Evaluation metric for multi-class classification
        n_estimators=100, 
        random_state=42,
        verbose=-1 # Suppress verbose output during training
    )
    model.fit(X_train, y_train)
    logger.info("LightGBM model training completed.")
    
    # --- Feature Importance Analysis ---
    importances = model.feature_importances_
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    logger.info("Feature Importances:")
    logger.info("\n" + feature_importance_df.to_string())
    
    print("\nFeature Importances:")
    print(feature_importance_df)
    # --- End of Feature Importance Analysis ---

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    logger.info(f"LightGBM Model Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")

    print(f"\nLightGBM Model Accuracy: {accuracy:.4f}")
    
    # Prepare preprocessing pipeline for return/saving
    preprocessing_pipeline = {'label_encoders': label_encoders}

    # print which label means what using label encoders
    le = label_encoders.get(target_column, None)
    if le:
        # Create a mapping from original label to encoded integer
        model_label_map = dict(zip(le.classes_, range(len(le.classes_))))
        # Create a reverse mapping from encoded integer back to original label
        rev_model_label_map = {v: k for k, v in model_label_map.items()}
        print(f"Label mapping (original -> encoded): {model_label_map}")
        print(f"Reverse label mapping (encoded -> original): {rev_model_label_map}")
    else:
        print("Target column was not encoded by LabelEncoder.")
        # If target column wasn't encoded, assume mapping is direct (value maps to itself)
        model_label_map = {label: label for label in y.unique()} 
        rev_model_label_map = {label: label for label in y.unique()} 


    dataset = get_dataset(dataset_type)
    combined_pred_by_idx = load_combined_pred_by_idx()
    
    # Convert predictions and true labels back to original format for scoring
    y_pred_df = pd.DataFrame(list(y_pred), columns=[target_column])
    y_test_df = pd.DataFrame(y_test, columns=[target_column])
    y_pred_df['idx'] = X_test.index
    y_test_df['idx'] = X_test.index
    y_pred_df[target_column] = y_pred_df[target_column].map(lambda x: rev_model_label_map.get(x, x))  # Map predictions back to original labels
    y_test_df[target_column] = y_test_df[target_column].map(lambda x: rev_model_label_map.get(x, x))  # Map test labels back to original labels
    
    
    for model_name, label_int in model_label_map.items():
        print(f"If only '{model_name}' is used (mapped to integer {label_int})")
        temp_y_pred_df = y_pred_df.copy() 
        temp_y_pred_df[target_column] = model_name
        score = get_combined_score(dataset, combined_pred_by_idx, temp_y_pred_df)
        print(f"Score for using only '{model_name}': {score:.4f}")
    
    score_router = get_combined_score(dataset, combined_pred_by_idx, y_pred_df)
    print(f"Score using LightGBM router: {score_router:.4f}")
    
    score_max_test = get_combined_score(dataset, combined_pred_by_idx, y_test_df)
    print(f"Max Score on test set (using true labels): {score_max_test:.4f}")

    return model, preprocessing_pipeline

if __name__ == '__main__':
    data_path = 'master_dataset_combined.csv'  # Replace with your dataset path
    target_column = 'model'  # Replace with your target column name
    
    trained_model, preprocessing_data = train_lightgbm_model(data_path, target_column, DatasetType.MMDD)
    
    if trained_model and preprocessing_data:
        # Save the model and preprocessing data (label encoders)
        joblib.dump(trained_model, 'lightgbm_model.pkl')
        logger.info("LightGBM model saved to lightgbm_model.pkl")
        joblib.dump(preprocessing_data, 'lightgbm_preprocessing_pipeline.pkl')
        logger.info("Preprocessing pipeline (label encoders) saved to lightgbm_preprocessing_pipeline.pkl")
        print("\nLightGBM model and preprocessing pipeline saved successfully.")
    else:
        logger.error("LightGBM model training failed or returned None. Model and preprocessing pipeline not saved.")
        print("\nLightGBM model training failed. Check logs for details. Model and preprocessing pipeline not saved.")