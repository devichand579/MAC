import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.preprocessing import LabelEncoder
import logging
import joblib
import argparse
import json
from typing import Dict, Any, Tuple, Optional, List

from router.utils.dataset_utils import DatasetType, get_combined_score, get_dataset, partial_f1
from router.utils.modelling_utils import load_combined_pred_by_idx # Moved import to the top

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
np.random.seed(42)


# Define cost mapping (avg time per sample in seconds)
COST_MAPPING = {
    'paligemma': 1.4802,
    'qwen2.vl.2b.instruct': 0.7338,
    'minicpm_image': 2.0891,
    'qb': 0.00094
}


def calculate_cost_score(y_pred: np.ndarray, rev_model_label_map: Dict[Any, str]) -> float:
    """
    Calculate the average cost per sample based on predictions.
    Lower cost is better, so we return negative cost for maximization.
    """
    total_cost = 0.0
    for pred_label in y_pred:
        model_name = rev_model_label_map.get(pred_label, None)
        if model_name:
            model_name_normalized = model_name.lower().strip()
            if model_name_normalized in COST_MAPPING:
                total_cost += COST_MAPPING[model_name_normalized]
            else:
                # Default cost if not found
                default_cost = sum(COST_MAPPING.values()) / len(COST_MAPPING)
                total_cost += default_cost
    
    avg_cost = total_cost / len(y_pred) if len(y_pred) > 0 else float('inf')
    return -avg_cost  # Negative because we want to maximize (lower cost is better)


def create_combined_scorer(rev_model_label_map: Dict[Any, str], 
                           cost_weight: float = 0.5) -> callable:
    """
    Create a custom scorer that combines accuracy and cost.
    
    Args:
        rev_model_label_map: Mapping from encoded labels to model names
        cost_weight: Weight for cost (0-1). Higher means more emphasis on cost.
    
    Returns:
        A scorer function that can be used with sklearn
    """
    def combined_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate accuracy (normalized to 0-1)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate cost score
        cost_score = calculate_cost_score(y_pred, rev_model_label_map)
        # Normalize cost score to 0-1 range (assuming max cost is around 2.1)
        max_cost = max(COST_MAPPING.values())
        normalized_cost = -cost_score / max_cost  # cost_score is negative, so negate
        
        # Combine performance and cost
        # Higher is better, so we want high performance and low cost (high normalized_cost)
        combined_score = (1 - cost_weight) * accuracy + cost_weight * normalized_cost
        
        return combined_score
    
    return make_scorer(combined_scorer, greater_is_better=True)


def train_random_forest_model(
    data_path: str, 
    target_column: str, 
    dataset_type: DatasetType,
    use_hyperparameter_tuning: bool = False,
    cost_weight: float = 0.5,
    n_iter: int = 50,
    cv: int = 3
):
    """
    Train a Random Forest model on the dataset located at data_path.
    Args:
        data_path (str): Path to the dataset CSV file.
        target_column (str): The column name in the dataset to predict.
        dataset_type (DatasetType): The type of dataset being used.
        use_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
        cost_weight (float): Weight for cost in combined score (0-1). Higher means more emphasis on cost.
        n_iter (int): Number of iterations for RandomizedSearchCV.
        cv (int): Number of cross-validation folds.
    Returns:
        model (RandomForestClassifier): The trained Random Forest model.
        label_encoders (dict): A dictionary of label encoders for categorical variables.
    """
    # Load the dataset
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} does not exist.")
        if use_hyperparameter_tuning:
            return None
        return None, None # Return two Nones as the original call expects two values
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        if use_hyperparameter_tuning:
            return None
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
        if use_hyperparameter_tuning:
            return None
        return None, None
        
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataset after preprocessing.")
        logger.error(f"Available columns: {df.columns.tolist()}")
        if use_hyperparameter_tuning:
            return None
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
        if use_hyperparameter_tuning:
            return None
        return None, None

    # Load dataset and combined predictions early (needed for custom scorer)
    dataset = get_dataset(dataset_type)
    combined_pred_by_idx_filename = f"combined_pred_by_idx_{dataset_type.value}.json"
    combined_pred_by_idx = load_combined_pred_by_idx(combined_pred_by_idx_filename)
    
    # Create reverse label mapping for scorer (needed before train_test_split for CV)
    le_target = label_encoders.get(target_column, None)
    if le_target is None:
        logger.error(f"Label encoder for target column '{target_column}' not found.")
        if use_hyperparameter_tuning:
            return None
        return None, None
    model_label_map = dict(zip(le_target.classes_, range(len(le_target.classes_))))
    rev_model_label_map = {v: k for k, v in model_label_map.items()}

    # set seed = 42

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Create and train the Random Forest model
    if use_hyperparameter_tuning:
        logger.info(f"Starting hyperparameter tuning with cost_weight={cost_weight}, n_iter={n_iter}, cv={cv}")
        
        # Define hyperparameter search space
        param_distributions = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
        
        # Create custom scorer
        custom_scorer = create_combined_scorer(
            rev_model_label_map=rev_model_label_map,
            cost_weight=cost_weight
        )
        
        # Create base model
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=custom_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        
        logger.info(f"Best hyperparameters: {random_search.best_params_}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
        print(f"\nBest hyperparameters: {random_search.best_params_}")
        print(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        # Extract all results from RandomizedSearchCV and calculate metrics
        logger.info("Extracting all trial results...")
        all_results = []
        cv_results = random_search.cv_results_
        
        # Create a validation set from training data for metric calculation
        # Note: We need to preserve the original 'idx' column for matching with combined_pred_by_idx
        # The 'idx' column should still be in the dataframe even though it's set as index
        X_train_for_val, X_val, y_train_for_val, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, 
            stratify=y_train if y_train.nunique() > 1 else None
        )
        
        # Get the idx column values - check if 'idx' is still a column or if we need to use index
        if 'idx' in X_val.columns:
            val_idx_values = X_val['idx'].values
        else:
            # idx is the index, get it directly
            val_idx_values = X_val.index.values
        
        # Reuse dataset and combined_pred_by_idx that were already loaded earlier
        # (they are available in the function scope)
        
        # Process each trial
        logger.info(f"Processing {len(cv_results['params'])} trials to calculate detailed metrics...")
        for i in range(len(cv_results['params'])):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing trial {i+1}/{len(cv_results['params'])}...")
            
            params = cv_results['params'][i]
            cv_score = cv_results['mean_test_score'][i]
            
            # Train model with these parameters on training set
            trial_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            trial_model.fit(X_train_for_val, y_train_for_val)
            
            # Predict on validation set
            y_val_pred = trial_model.predict(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_val_pred)
            
            # Calculate avg_cost
            total_cost = 0.0
            for pred_label in y_val_pred:
                model_name = rev_model_label_map.get(pred_label, None)
                if model_name:
                    model_name_normalized = model_name.lower().strip()
                    if model_name_normalized in COST_MAPPING:
                        total_cost += COST_MAPPING[model_name_normalized]
                    else:
                        default_cost = sum(COST_MAPPING.values()) / len(COST_MAPPING)
                        total_cost += default_cost
            avg_cost = total_cost / len(y_val_pred) if len(y_val_pred) > 0 else float('inf')
            
            # Calculate combined score
            # Filter to only include indices that exist in combined_pred_by_idx
            valid_indices = []
            valid_predictions = []
            
            # Debug: Check a few sample indices
            if i == 0 and len(val_idx_values) > 0:
                sample_indices = list(val_idx_values[:3])
                sample_combined_keys = list(combined_pred_by_idx.keys())[:3] if combined_pred_by_idx else []
                logger.info(f"Sample validation indices: {sample_indices} (types: {[type(x).__name__ for x in sample_indices]})")
                logger.info(f"Sample combined_pred_by_idx keys: {sample_combined_keys} (types: {[type(x).__name__ for x in sample_combined_keys]})")
                logger.info(f"Total validation samples: {len(val_idx_values)}, Total combined_pred keys: {len(combined_pred_by_idx)}")
                # Check if any match
                matches = sum(1 for idx in val_idx_values[:10] if str(idx) in combined_pred_by_idx or idx in combined_pred_by_idx)
                logger.info(f"Matches in first 10: {matches}/10")
            
            for idx, pred in zip(val_idx_values, y_val_pred):
                # Try multiple formats
                idx_str = str(idx)
                # Check if string version exists
                if idx_str in combined_pred_by_idx:
                    valid_indices.append(idx)
                    valid_predictions.append(pred)
                # Try with the index as-is (in case it's already the right format)
                elif idx in combined_pred_by_idx:
                    valid_indices.append(idx)
                    valid_predictions.append(pred)
            
            combined_score = None
            if len(valid_indices) > 0:
                y_val_pred_df = pd.DataFrame({
                    'idx': valid_indices,
                    'model': [rev_model_label_map.get(pred, 'unknown') for pred in valid_predictions]
                })
                try:
                    combined_score = get_combined_score(dataset, combined_pred_by_idx, y_val_pred_df)
                except Exception as e:
                    logger.warning(f"Could not calculate combined score for trial {i+1}: {e}")
            else:
                if i == 0:  # Only log detailed info for first trial
                    logger.warning(f"No valid indices found in combined_pred_by_idx for trial {i+1}")
                    logger.warning(f"First 5 validation indices: {list(val_idx_values[:5])}")
                    logger.warning(f"First 5 combined_pred keys: {list(combined_pred_by_idx.keys())[:5] if len(combined_pred_by_idx) > 0 else 'empty'}")
                else:
                    logger.warning(f"No valid indices found in combined_pred_by_idx for trial {i+1}")
            
            # Calculate the cost-weighted score (same as CV score)
            cost_score = -avg_cost  # Negative because lower cost is better
            max_cost = max(COST_MAPPING.values())
            normalized_cost = -cost_score / max_cost
            score = (1 - cost_weight) * accuracy + cost_weight * normalized_cost
            
            result = {
                'trial': i + 1,
                'params': params,
                'cv_score': float(cv_score),
                'metrics': {
                    'accuracy': float(accuracy),
                    'avg_cost': float(avg_cost),
                    'combined_score': combined_score
                },
                'score': float(score)
            }
            all_results.append(result)
        
        # Find best result
        best_result = max(all_results, key=lambda x: x['score'])
        
        # Return results for this cost_weight
        return {
            'cost_weight': cost_weight,
            'best_params': random_search.best_params_,
            'best_cv_score': float(random_search.best_score_),
            'best_result': {
                'trial': best_result['trial'],
                'params': best_result['params'],
                'score': best_result['score'],
                'metrics': best_result['metrics']
            },
            'all_results': all_results,
            'model': model,
            'label_encoders': label_encoders
        }
    else:
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
    print(f"label mapping: {model_label_map}")
    y_pred_df = pd.DataFrame(list(y_pred), columns=[target_column])
    y_test_df = pd.DataFrame(y_test, columns=[target_column])
    y_pred_df['idx'] = X_test.index
    y_test_df['idx'] = X_test.index
    y_pred_df[target_column] = y_pred_df[target_column].map(lambda x: rev_model_label_map.get(x, x))  # Map predictions back to original labels
    pred_counts = y_pred_df[target_column].value_counts()
    print(f"\nPredicted model distribution:")
    print(pred_counts)
    
    # Calculate final time cost based on prediction fractions
    total_test_samples = len(y_pred_df)
    final_time_per_sample = 0.0
    total_time = 0.0
    
    print(f"\nCost analysis for test set:")
    print(f"{'Model':<30} {'Count':<10} {'Fraction':<12} {'Cost (s/sample)':<18} {'Fraction×Cost':<15} {'Total Time (s)':<15}")
    print("-" * 100)
    
    for model_name in pred_counts.index:
        count = pred_counts[model_name]
        fraction = count / total_test_samples
        # Get cost for this model (normalize name to match COST_MAPPING keys)
        model_name_normalized = model_name.lower().strip()
        if model_name_normalized in COST_MAPPING:
            cost = COST_MAPPING[model_name_normalized]
        else:
            # Default cost if not found
            default_cost = sum(COST_MAPPING.values()) / len(COST_MAPPING)
            cost = default_cost
            logger.warning(f"Model '{model_name}' not found in COST_MAPPING. Using default cost: {default_cost}")
        
        fraction_times_cost = fraction * cost
        model_total_time = count * cost
        final_time_per_sample += fraction_times_cost
        total_time += model_total_time
        
        print(f"{model_name:<30} {count:<10} {fraction:<12.4f} {cost:<18.4f} {fraction_times_cost:<15.4f} {model_total_time:<15.4f}")
    
    print("-" * 100)
    print(f"{'SUM (Average Time per Sample)':<30} {'-':<10} {'-':<12} {'-':<18} {final_time_per_sample:<15.4f} {'-':<15}")
    print(f"{'Total Time for Test Set':<30} {'-':<10} {'-':<12} {'-':<18} {'-':<15} {total_time:<15.4f}")
    print(f"\nFinal average time per sample (sum of fraction × cost): {final_time_per_sample:.4f} seconds")
    print(f"Total time for test set ({total_test_samples} samples): {total_time:.4f} seconds")
    
    logger.info(f"Final average time per sample: {final_time_per_sample:.4f} seconds")
    logger.info(f"Total time for test set: {total_time:.4f} seconds")
    
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
    
    # Return format depends on whether hyperparameter tuning was used
    if use_hyperparameter_tuning:
        # This should have been returned earlier, but just in case
        return {
            'cost_weight': cost_weight,
            'best_params': None,
            'best_cv_score': None,
            'best_result': None,
            'all_results': [],
            'model': model,
            'label_encoders': label_encoders
        }
    else:
        return model, label_encoders

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='master_dataset_combined.csv')
    parser.add_argument('--dataset', type=str, default='mmdd', choices=['mmdd', 'image_chat'])
    parser.add_argument('--use_hyperparameter_tuning', action='store_true', 
                       help='Enable hyperparameter tuning with cost-accuracy tradeoff')
    parser.add_argument('--cost_weight', type=float, default=None,
                       help='Weight for cost in combined score (0-1). If not specified and use_hyperparameter_tuning is True, will run for all weights [0.0, 0.25, 0.5, 0.75, 1.0]')
    parser.add_argument('--n_iter', type=int, default=50,
                       help='Number of iterations for RandomizedSearchCV. Default: 50')
    parser.add_argument('--cv', type=int, default=3,
                       help='Number of cross-validation folds. Default: 3')
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
    
    # Define cost weights to test
    if args.use_hyperparameter_tuning:
        if args.cost_weight is not None:
            # Single cost_weight specified
            if args.cost_weight < 0 or args.cost_weight > 1:
                raise ValueError(f"cost_weight must be between 0 and 1, got {args.cost_weight}")
            cost_weights = [args.cost_weight]
        else:
            # Run for all cost weights automatically
            cost_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
            logger.info(f"Running hyperparameter tuning for all cost weights: {cost_weights}")
            print(f"\nRunning hyperparameter tuning for all cost weights: {cost_weights}")
    else:
        # Not using hyperparameter tuning, use default or specified cost_weight
        cost_weights = [args.cost_weight if args.cost_weight is not None else 0.5]
    
    # Collect results for all cost weights
    all_cost_weight_results = {}
    best_model_overall = None
    best_encoders = None
    best_score_overall = float('-inf')
    
    for cost_weight in cost_weights:
        logger.info(f"\n{'='*80}")
        logger.info(f"TUNING FOR COST_WEIGHT = {cost_weight}")
        logger.info(f"{'='*80}")
        print(f"\n{'='*80}")
        print(f"TUNING FOR COST_WEIGHT = {cost_weight}")
        print(f"{'='*80}")
        
        result = train_random_forest_model(
            data_path, 
            target_column, 
            dataset,
            use_hyperparameter_tuning=args.use_hyperparameter_tuning,
            cost_weight=cost_weight,
            n_iter=args.n_iter,
            cv=args.cv
        )
        
        if result is None:
            logger.warning(f"No result returned for cost_weight={cost_weight}")
            continue
        
        # Check if result is a tuple (old format) or dict (new format)
        if isinstance(result, tuple):
            logger.error(f"Unexpected tuple return for cost_weight={cost_weight}. Expected dict when use_hyperparameter_tuning=True.")
            continue
        
        # Store results for this cost_weight
        all_cost_weight_results[cost_weight] = {
            'best_params': result['best_params'],
            'best_cv_score': result['best_cv_score'],
            'best_result': result['best_result'],
            'all_results': result['all_results']
        }
        
        # Track best model overall
        if result['best_result']['score'] > best_score_overall:
            best_score_overall = result['best_result']['score']
            best_model_overall = result['model']
            best_encoders = result['label_encoders']
    
    # Save combined results to JSON (similar to neural classifier format)
    if args.use_hyperparameter_tuning and len(cost_weights) > 1:
        results_file = f'rf_hyperparameter_{dataset.value}.json'
        output_data = {
            'n_iter': args.n_iter,
            'cv': args.cv,
            'best_per_cost_weight': {
                str(cw): {
                    'params': all_cost_weight_results[cw]['best_params'],
                    'score': all_cost_weight_results[cw]['best_result']['score'],
                    'metrics': all_cost_weight_results[cw]['best_result']['metrics']
                }
                for cw in sorted(all_cost_weight_results.keys())
            },
            'all_results': []
        }
        
        # Combine all_results from all cost weights
        for cw in sorted(all_cost_weight_results.keys()):
            for trial_result in all_cost_weight_results[cw]['all_results']:
                trial_result['cost_weight'] = cw
                output_data['all_results'].append(trial_result)
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\nAll results saved to {results_file}")
        print(f"\nAll results saved to {results_file}")
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("HYPERPARAMETER TUNING SUMMARY - BEST MODEL PER COST WEIGHT")
        logger.info(f"{'='*80}")
        for cw in sorted(all_cost_weight_results.keys()):
            best = all_cost_weight_results[cw]['best_result']
            logger.info(f"\nCost Weight: {cw}")
            logger.info(f"  Best Score: {best['score']:.4f}")
            logger.info(f"  Accuracy: {best['metrics']['accuracy']:.4f}")
            logger.info(f"  Avg Cost: {best['metrics']['avg_cost']:.4f}")
            logger.info(f"  Params: {best['params']}")
    
    # Save the best model overall
    if best_model_overall and best_encoders:
        joblib.dump(best_model_overall, 'random_forest_model.pkl')
        logger.info("Best model saved to random_forest_model.pkl")
        joblib.dump(best_encoders, 'label_encoders.pkl')
        logger.info("Label encoders saved to label_encoders.pkl")
        print("\nBest model and label encoders saved successfully.")
    elif not args.use_hyperparameter_tuning:
        # If not using hyperparameter tuning, save the single model
        trained_model, encoders = train_random_forest_model(
            data_path, 
            target_column, 
            dataset,
            use_hyperparameter_tuning=False,
            cost_weight=cost_weights[0],
            n_iter=args.n_iter,
            cv=args.cv
        )
        if trained_model and encoders:
            joblib.dump(trained_model, 'random_forest_model.pkl')
            logger.info("Model saved to random_forest_model.pkl")
            joblib.dump(encoders, 'label_encoders.pkl')
            logger.info("Label encoders saved to label_encoders.pkl")
            print("\nModel and label encoders saved successfully.")
        else:
            logger.error("Model training failed or returned None. Model and encoders not saved.")
            print("\nModel training failed. Check logs for details. Model and encoders not saved.")
