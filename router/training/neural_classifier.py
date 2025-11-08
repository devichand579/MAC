import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import logging
import joblib
import argparse
from typing import Dict, Any, Tuple, Optional, List
import json
import random

from router.utils.dataset_utils import DatasetType, get_combined_score, get_dataset, partial_f1
from router.utils.modelling_utils import load_combined_pred_by_idx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)





class EmbeddingDataset(Dataset):
    """Dataset class for embeddings and labels"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """2-3 layer MLP for model classification"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_mlp_model(embeddings_path: str, mapping_path: str, dataset_type: DatasetType, 
                   hidden_dims=[256, 128, 64], epochs=100, batch_size=256, learning_rate=0.001, 
                   device='cuda' if torch.cuda.is_available() else 'cpu', 
                   use_cost_loss=False, lambda_weight=0.5, dropout=0.2, 
                   return_val_metrics=False, val_split=0.2):
    """
    Train a 2-3 layer MLP on embeddings for model classification.
    
    Args:
        embeddings_path: Path to the embeddings .npy file
        mapping_path: Path to the embedding mapping CSV file
        dataset_type: The type of dataset being used
        hidden_dims: List of hidden layer dimensions (default: [128, 64] for 2 hidden layers)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
        use_cost_loss: Whether to use cost-weighted loss (default: False)
        lambda_weight: Weight for cross-entropy loss in combined loss: lambda * CE + (1-lambda) * Cost (default: 0.5)
        dropout: Dropout rate (default: 0.2)
        return_val_metrics: If True, return validation metrics instead of full evaluation (default: False)
        val_split: Validation split ratio (default: 0.2)
    
    Returns:
        model: Trained PyTorch model
        label_encoder: LabelEncoder for model labels
        (if return_val_metrics=True) metrics dict with 'accuracy', 'combined_score', 'avg_cost'
    """
    # Load embeddings and mapping
    logger.info(f"Loading embeddings from {embeddings_path}")
    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings file {embeddings_path} does not exist.")
        return None, None
    
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    
    logger.info(f"Loading mapping from {mapping_path}")
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping file {mapping_path} does not exist.")
        return None, None
    
    mapping_df = pd.read_csv(mapping_path)
    logger.info(f"Loaded mapping with shape: {mapping_df.shape}")
    
    # Ensure embeddings and mapping are aligned by embedding_index
    mapping_df = mapping_df.sort_values('embedding_index').reset_index(drop=True)
    
    # Extract labels
    model_labels = mapping_df['model'].values
    idx_values = mapping_df['idx'].values
    
    # Encode model labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(model_labels)
    num_classes = len(label_encoder.classes_)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Model label distribution:\n{pd.Series(model_labels).value_counts().to_string()}")
    
    # Define cost mapping (avg time per sample in seconds)
    cost_mapping = {
        'paligemma': 1.4802,
        'qwen2.vl.2b.instruct': 0.7338,
        'minicpm_image': 2.0891,
        'qb': 0.00094
    }
    
    # Create cost tensor for each class
    cost_tensor = None
    if use_cost_loss:
        costs = []
        for class_name in label_encoder.classes_:
            # Normalize class name to match cost_mapping keys
            class_name_normalized = class_name.lower().strip()
            if class_name_normalized in cost_mapping:
                costs.append(cost_mapping[class_name_normalized])
            else:
                # Default cost if model not in mapping (use average)
                default_cost = sum(cost_mapping.values()) / len(cost_mapping)
                logger.warning(f"Model '{class_name}' not found in cost_mapping. Using default cost: {default_cost}")
                costs.append(default_cost)
        
        # Normalize costs to [0, 1] range using max normalization (aligned with RandomForest)
        max_cost = max(cost_mapping.values())  # Use max from cost_mapping, not from costs
        if max_cost > 0:
            costs_normalized = [c / max_cost for c in costs]
        else:
            costs_normalized = [0.5] * len(costs)  # All same cost
        
        cost_tensor = torch.FloatTensor(costs_normalized).to(device)
        logger.info(f"Cost tensor (normalized): {dict(zip(label_encoder.classes_, costs_normalized))}")
        logger.info(f"Using cost-weighted loss with lambda={lambda_weight}")
    
    # Verify embeddings and labels are aligned
    if len(embeddings) != len(encoded_labels):
        logger.error(f"Mismatch: embeddings length ({len(embeddings)}) != labels length ({len(encoded_labels)})")
        # Align them using embedding_index
        max_idx = min(len(embeddings), len(mapping_df))
        embeddings = embeddings[:max_idx]
        encoded_labels = encoded_labels[:max_idx]
        idx_values = idx_values[:max_idx]
        logger.info(f"Aligned to {max_idx} samples")
    
    # Split data
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        embeddings, encoded_labels, idx_values, 
        test_size=val_split, random_state=42, 
        stratify=encoded_labels if len(np.unique(encoded_labels)) > 1 else None
    )
    
    logger.info(f"Training set shape: {X_train.shape}, Test/Val set shape: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(X_train, y_train)
    test_dataset = EmbeddingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = embeddings.shape[1]
    model = MLPClassifier(input_dim, hidden_dims, num_classes, dropout=dropout).to(device)
    logger.info(f"Model architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info(f"Starting training on {device}...")
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            
            # Compute cross-entropy loss
            ce_loss = criterion(outputs, batch_labels)
            
            # Compute combined loss if cost loss is enabled
            if use_cost_loss and cost_tensor is not None:
                # Get softmax probabilities
                probs = torch.softmax(outputs, dim=1)
                # Compute expected cost: sum over classes of (probability * cost)
                expected_cost = torch.sum(probs * cost_tensor.unsqueeze(0), dim=1)
                # Average over batch
                cost_loss = torch.mean(expected_cost)
                # Combined loss: lambda * CE + (1 - lambda) * Cost
                loss = (1-lambda_weight) * ce_loss + lambda_weight * cost_loss
            else:
                loss = ce_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_labels in test_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_embeddings)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    logger.info(f"Training completed. Best test accuracy: {best_test_acc:.2f}%")
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            outputs = model(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Create reverse mapping for model labels
    model_label_map = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    rev_model_label_map = {v: k for k, v in model_label_map.items()}
    
    # If return_val_metrics is True, return early with validation metrics
    if return_val_metrics:
        # Calculate average cost
        avg_cost = 0.0
        for pred in all_preds:
            model_name = rev_model_label_map[pred]
            model_name_normalized = model_name.lower().strip()
            if model_name_normalized in cost_mapping:
                avg_cost += cost_mapping[model_name_normalized]
            else:
                avg_cost += sum(cost_mapping.values()) / len(cost_mapping)
        avg_cost /= len(all_preds) if len(all_preds) > 0 else 1.0
        
        # Calculate combined score if dataset is available
        combined_score = None
        try:
            dataset = get_dataset(dataset_type)
            combined_pred_by_idx_filename = f"combined_pred_by_idx_{dataset_type.value}.json"
            combined_pred_by_idx = load_combined_pred_by_idx(combined_pred_by_idx_filename)
            
            y_pred_df = pd.DataFrame({
                'idx': idx_test,
                'model': [rev_model_label_map[pred] for pred in all_preds]
            })
            combined_score = get_combined_score(dataset, combined_pred_by_idx, y_pred_df)
        except Exception as e:
            logger.warning(f"Could not calculate combined score: {e}")
        
        metrics = {
            'accuracy': accuracy,
            'avg_cost': avg_cost,
            'combined_score': combined_score
        }
        return model, label_encoder, metrics
    
    # Full evaluation (original behavior)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    logger.info(f"Final Test Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")
    
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Label mapping: {model_label_map}")
    
    # Evaluate with combined score
    dataset = get_dataset(dataset_type)
    combined_pred_by_idx_filename = f"combined_pred_by_idx_{dataset_type.value}.json"
    combined_pred_by_idx = load_combined_pred_by_idx(combined_pred_by_idx_filename)
    
    y_pred_df = pd.DataFrame({
        'idx': idx_test,
        'model': [rev_model_label_map[pred] for pred in all_preds]
    })
    y_test_df = pd.DataFrame({
        'idx': idx_test,
        'model': [rev_model_label_map[label] for label in all_labels]
    })
    
    print(f"\nPredicted model distribution:")
    pred_counts = y_pred_df['model'].value_counts()
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
        # Get cost for this model (normalize name to match cost_mapping keys)
        model_name_normalized = model_name.lower().strip()
        if model_name_normalized in cost_mapping:
            cost = cost_mapping[model_name_normalized]
        else:
            # Default cost if not found
            default_cost = sum(cost_mapping.values()) / len(cost_mapping)
            cost = default_cost
        
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
    
    for model_name, label in model_label_map.items():
        print(f"If only {model_name} is used")
        temp_y_pred_df = y_pred_df.copy()
        temp_y_pred_df['model'] = model_name
        score = get_combined_score(dataset, combined_pred_by_idx, temp_y_pred_df)
        print(f"Score for {model_name}: {score}")
    
    score = get_combined_score(dataset, combined_pred_by_idx, y_pred_df)
    print(f"Score using router: {score}")
    score = get_combined_score(dataset, combined_pred_by_idx, y_test_df)
    print(f"Max Score on test set: {score}")
    
    return model, label_encoder


def hyperparameter_tuning(
    embeddings_path: str,
    mapping_path: str,
    dataset_type: DatasetType,
    scoring_metric: str = 'cost_weighted',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    val_split: float = 0.2,
    n_iter: int = 50
) -> Tuple[Dict[str, Any], Any, Any]:
    """
    Perform hyperparameter tuning for MLP classifier using random search.
    
    Args:
        embeddings_path: Path to the embeddings .npy file
        mapping_path: Path to the embedding mapping CSV file
        dataset_type: The type of dataset being used
        scoring_metric: Metric to optimize (always 'cost_weighted' for cost-accuracy tradeoff)
        device: Device to use ('cuda' or 'cpu')
        val_split: Validation split ratio (default: 0.2)
        n_iter: Number of random hyperparameter combinations to try (default: 50)
    
    Returns:
        best_params: Dictionary of best hyperparameters
        best_model: Best trained model
        best_label_encoder: Label encoder for best model
    """
    # Always use cost_weighted scoring
    scoring_metric = 'cost_weighted'
    logger.info(f"Starting hyperparameter tuning with random search (n_iter={n_iter}), metric={scoring_metric} (always cost_weighted)")
    
    # Define search space - only tuning: hidden_dims, epochs, learning_rate, lambda_weight
    # Using default values for batch_size=256 and dropout=0.2
    search_space = {
        'hidden_dims': [
            [128, 64],           # 2 layers
            [256, 128],           # 2 layers
            [256, 128, 64],       # 3 layers
            [512, 256],           # 2 layers
            [512, 256, 128],      # 3 layers
            [128],                # 1 layer
            [256],                # 1 layer
            [64, 32],             # 2 layers (smaller)
        ],
        'epochs': [50, 100],
        'learning_rate': [0.0001, 0.0005, 0.001],
        'lambda_weight': [0.0, 0.25, 0.5, 0.75, 1.0]  # Always use cost_weighted, so always include lambda options
    }
    
    # Default values for non-tuned parameters
    default_batch_size = 256
    default_dropout = 0.2
    
    # Cost mapping for normalization (same as RandomForest)
    cost_mapping = {
        'paligemma': 1.4802,
        'qwen2.vl.2b.instruct': 0.7338,
        'minicpm_image': 2.0891,
        'qb': 0.00094
    }
    max_cost = max(cost_mapping.values())  # 2.0891
    
    # Track best model for each lambda weight separately
    best_per_lambda = {}  # {lambda_weight: {'params': ..., 'model': ..., 'label_encoder': ..., 'score': ..., 'metrics': ...}}
    all_results = []
    
    # Generate hyperparameter combinations using random search
    # Do n_iter iterations for each lambda_weight value
    all_lambda_weights = search_space['lambda_weight']
    total_trials = len(all_lambda_weights) * n_iter
    logger.info(f"Generated {total_trials} random hyperparameter combinations to try")
    logger.info(f"({n_iter} iterations for each of {len(all_lambda_weights)} lambda_weight values)")
    logger.info(f"Will find best model separately for each lambda_weight: {all_lambda_weights}")
    
    # Run trials - process each lambda weight separately
    for lambda_weight in all_lambda_weights:
        logger.info(f"\n{'='*80}")
        logger.info(f"TUNING FOR LAMBDA_WEIGHT = {lambda_weight}")
        logger.info(f"{'='*80}")
        
        best_score = float('-inf')
        best_params = None
        best_model = None
        best_label_encoder = None
        best_metrics = None
        
        # Generate n_iter random combinations for this lambda_weight
        trials = []
        for _ in range(n_iter):
            trial = {
                'hidden_dims': random.choice(search_space['hidden_dims']),
                'epochs': random.choice(search_space['epochs']),
                'learning_rate': random.choice(search_space['learning_rate']),
                'use_cost_loss': True,
                'lambda_weight': lambda_weight  # Fixed for this lambda group
            }
            trials.append(trial)
        
        # Run trials for this lambda_weight
        for i, params in enumerate(trials):
            logger.info(f"\n{'-'*80}")
            logger.info(f"Lambda {lambda_weight} - Trial {i+1}/{len(trials)}")
            logger.info(f"Hyperparameters: {params}")
            logger.info(f"{'-'*80}")
            
            try:
                # Train model with these hyperparameters (using defaults for batch_size and dropout)
                model, label_encoder, metrics = train_mlp_model(
                    embeddings_path=embeddings_path,
                    mapping_path=mapping_path,
                    dataset_type=dataset_type,
                    hidden_dims=params['hidden_dims'],
                    epochs=params['epochs'],
                    batch_size=default_batch_size,  # Use default
                    learning_rate=params['learning_rate'],
                    device=device,
                    use_cost_loss=params['use_cost_loss'],
                    lambda_weight=params['lambda_weight'],
                    dropout=default_dropout,  # Use default
                    return_val_metrics=True,
                    val_split=val_split
                )
                
                if model is None or label_encoder is None:
                    logger.warning(f"Lambda {lambda_weight} - Trial {i+1} failed: model training returned None")
                    continue
                
                # Use cost_weighted scoring matching RandomForest approach exactly
                # RandomForest: 
                #   1. calculate_cost_score returns -avg_cost (negative)
                #   2. normalized_cost = -cost_score / max_cost = avg_cost / max_cost (positive, 0-1)
                #   3. combined_score = (1 - cost_weight) * accuracy + cost_weight * normalized_cost
                # For neural: we use lambda_weight as cost_weight (weight for cost in score)
                lambda_w = params['lambda_weight']  # This acts as cost_weight in the scorer
                # Calculate cost_score as negative (matching RandomForest)
                cost_score = -metrics['avg_cost']  # Negative because lower cost is better
                # Normalize cost score to 0-1 range (matching RandomForest)
                normalized_cost = -cost_score / max_cost  # cost_score is negative, so negate to get positive
                # Combined score: (1 - lambda) * accuracy + lambda * normalized_cost
                # Higher is better for both accuracy and normalized_cost
                score = (1 - lambda_w) * metrics['accuracy'] + lambda_w * normalized_cost
                
                result = {
                    'trial': i + 1,
                    'lambda_weight': lambda_weight,
                    'params': params,
                    'metrics': metrics,
                    'score': score
                }
                all_results.append(result)
                
                logger.info(f"Lambda {lambda_weight} - Trial {i+1} Results:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Avg latency: {metrics['avg_cost']:.4f}")
                if metrics['combined_score'] is not None:
                    if isinstance(metrics['combined_score'], dict):
                        # get_combined_score returns a dict with f1, precision, recall, syntactic_match
                        logger.info(f"  Combined Score: {metrics['combined_score']}")
                    else:
                        logger.info(f"  Combined Score: {metrics['combined_score']:.4f}")
                logger.info(f"  Cost-Weighted Score: {score:.4f} ((1-{params['lambda_weight']:.2f}) * accuracy + {params['lambda_weight']:.2f} * normalized_cost)")
                
                # Update best for this lambda_weight if this is better
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_model = model
                    best_label_encoder = label_encoder
                    best_metrics = metrics.copy()
                    logger.info(f"  *** New best for lambda {lambda_weight}! ***")
                
            except Exception as e:
                logger.error(f"Lambda {lambda_weight} - Trial {i+1} failed with error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Store best model for this lambda_weight
        if best_params is not None:
            best_per_lambda[lambda_weight] = {
                'params': best_params,
                'model': best_model,
                'label_encoder': best_label_encoder,
                'score': best_score,
                'metrics': best_metrics
            }
            logger.info(f"\nBest model for lambda_weight={lambda_weight}:")
            logger.info(f"  Score: {best_score:.4f}")
            logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
            logger.info(f"  Avg Cost: {best_metrics['avg_cost']:.4f}")
            logger.info(f"  Params: hidden_dims={best_params['hidden_dims']}, epochs={best_params['epochs']}, lr={best_params['learning_rate']}")
        else:
            logger.warning(f"No successful trials for lambda_weight={lambda_weight}")
    
    # Print summary of best models per lambda
    logger.info(f"\n{'='*80}")
    logger.info("HYPERPARAMETER TUNING SUMMARY - BEST MODEL PER LAMBDA WEIGHT")
    logger.info(f"{'='*80}")
    for lambda_w in sorted(best_per_lambda.keys()):
        best = best_per_lambda[lambda_w]
        logger.info(f"\nLambda Weight: {lambda_w}")
        logger.info(f"  Best Score: {best['score']:.4f}")
        logger.info(f"  Accuracy: {best['metrics']['accuracy']:.4f}")
        logger.info(f"  Avg Cost: {best['metrics']['avg_cost']:.4f}")
        logger.info(f"  Params: {best['params']}")
    
    # Save results to JSON
    results_file = f'hyperparameter_tuning_results_{dataset_type.value}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_per_lambda': {
                str(lambda_w): {
                    'params': best_per_lambda[lambda_w]['params'],
                    'score': best_per_lambda[lambda_w]['score'],
                    'metrics': best_per_lambda[lambda_w]['metrics']
                }
                for lambda_w in best_per_lambda.keys()
            },
            'scoring_metric': scoring_metric,
            'all_results': [
                {
                    'trial': r['trial'],
                    'lambda_weight': r.get('lambda_weight', r['params']['lambda_weight']),
                    'params': r['params'],
                    'metrics': r['metrics'],
                    'score': r['score']
                }
                for r in all_results
            ]
        }, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    # Print performance summary table for best models per lambda
    if best_per_lambda:
        logger.info(f"\n{'='*80}")
        logger.info("PERFORMANCE SUMMARY - BEST MODEL FOR EACH LAMBDA WEIGHT")
        logger.info(f"{'='*80}")
        
        # Print summary table
        logger.info(f"{'Lambda':<10} {'Accuracy':<12} {'Avg Cost':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'Architecture':<30}")
        logger.info("-" * 100)
        for lambda_w in sorted(best_per_lambda.keys()):
            best = best_per_lambda[lambda_w]
            metrics = best['metrics']
            combined = metrics['combined_score']
            params = best['params']
            arch_str = f"{params['hidden_dims']}, {params['epochs']}ep, lr={params['learning_rate']}"
            
            if combined is not None and isinstance(combined, dict):
                f1 = combined.get('f1', 'N/A')
                precision = combined.get('precision', 'N/A')
                recall = combined.get('recall', 'N/A')
                logger.info(f"{lambda_w:<10.2f} {metrics['accuracy']:<12.4f} {metrics['avg_cost']:<12.4f} "
                          f"{f1:<12.4f if isinstance(f1, (int, float)) else 'N/A':<12} "
                          f"{precision:<12.4f if isinstance(precision, (int, float)) else 'N/A':<12} "
                          f"{recall:<12.4f if isinstance(recall, (int, float)) else 'N/A':<12} "
                          f"{arch_str:<30}")
            else:
                logger.info(f"{lambda_w:<10.2f} {metrics['accuracy']:<12.4f} {metrics['avg_cost']:<12.4f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {arch_str:<30}")
        
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY - BEST MODEL FOR EACH LAMBDA WEIGHT")
        print(f"{'='*80}")
        print(f"{'Lambda':<10} {'Accuracy':<12} {'Avg Cost':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'Architecture':<30}")
        print("-" * 100)
        for lambda_w in sorted(best_per_lambda.keys()):
            best = best_per_lambda[lambda_w]
            metrics = best['metrics']
            combined = metrics['combined_score']
            params = best['params']
            arch_str = f"{params['hidden_dims']}, {params['epochs']}ep, lr={params['learning_rate']}"
            
            if combined is not None and isinstance(combined, dict):
                f1 = combined.get('f1', 'N/A')
                precision = combined.get('precision', 'N/A')
                recall = combined.get('recall', 'N/A')
                f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else 'N/A'
                prec_str = f"{precision:.4f}" if isinstance(precision, (int, float)) else 'N/A'
                rec_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else 'N/A'
                print(f"{lambda_w:<10.2f} {metrics['accuracy']:<12.4f} {metrics['avg_cost']:<12.4f} "
                      f"{f1_str:<12} {prec_str:<12} {rec_str:<12} {arch_str:<30}")
            else:
                print(f"{lambda_w:<10.2f} {metrics['accuracy']:<12.4f} {metrics['avg_cost']:<12.4f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {arch_str:<30}")
    
    # Return the model with the highest score across ALL lambda weights
    if best_per_lambda:
        # Find the lambda weight with the highest score
        best_lambda = max(best_per_lambda.keys(), key=lambda l: best_per_lambda[l]['score'])
        best_params = best_per_lambda[best_lambda]['params']
        best_model = best_per_lambda[best_lambda]['model']
        best_label_encoder = best_per_lambda[best_lambda]['label_encoder']
        logger.info(f"\nOverall best model (highest score across all lambda weights):")
        logger.info(f"  Lambda weight: {best_lambda}")
        logger.info(f"  Score: {best_per_lambda[best_lambda]['score']:.4f}")
        logger.info(f"  Accuracy: {best_per_lambda[best_lambda]['metrics']['accuracy']:.4f}")
        logger.info(f"  Avg Cost: {best_per_lambda[best_lambda]['metrics']['avg_cost']:.4f}")
    else:
        best_params = None
        best_model = None
        best_label_encoder = None
    
    return best_params, best_model, best_label_encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP classifier on embeddings for model selection")
    parser.add_argument('--embeddings_path', type=str, default=None, 
                       help='Path to embeddings .npy file (default: auto-detect from dataset)')
    parser.add_argument('--mapping_path', type=str, default=None,
                       help='Path to embedding mapping CSV (default: auto-detect from dataset)')
    parser.add_argument('--dataset', type=str, default='mmdd', choices=['mmdd', 'image_chat'],
                       help='Dataset type')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[ 256, 128, 64],
                       help='Hidden layer dimensions (default: [128, 64] for 2 hidden layers)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--use_cost_loss', action='store_true',
                       help='Enable cost-weighted loss: lambda * CE + (1-lambda) * Cost')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_weight',
                       help='Weight for cross-entropy loss in combined loss (default: 0.5)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    parser.add_argument('--use_hyperparameter_tuning', action='store_true',
                       help='Enable hyperparameter tuning (uses random search)')
    parser.add_argument('--n_iter', type=int, default=50,
                       help='Number of random hyperparameter combinations to try (default: 50)')
    
    args = parser.parse_args()
    
    # Determine dataset type
    if args.dataset == "mmdd":
        dataset_type = DatasetType.MMDD
        default_embeddings = 'master_dataset_mmdd_embeddings.npy'
        default_mapping = 'master_dataset_mmdd_embedding_mapping.csv'
    elif args.dataset == "image_chat":
        dataset_type = DatasetType.IMAGECHAT
        default_embeddings = 'master_dataset_imagechat_embeddings.npy'
        default_mapping = 'master_dataset_imagechat_embedding_mapping.csv'
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Set default paths if not provided
    embeddings_path = args.embeddings_path or default_embeddings
    mapping_path = args.mapping_path or default_mapping
    
    # Auto-detect device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using embeddings: {embeddings_path}")
    logger.info(f"Using mapping: {mapping_path}")
    logger.info(f"Using device: {device}")
    
    if args.use_hyperparameter_tuning:
        logger.info("Starting hyperparameter tuning...")
        best_params, trained_model, label_encoder = hyperparameter_tuning(
            embeddings_path=embeddings_path,
            mapping_path=mapping_path,
            dataset_type=dataset_type,
            device=device,
            n_iter=args.n_iter
        )
        
        if trained_model and label_encoder:
            logger.info(f"\nBest hyperparameters found:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")
            
            # Train final model with best params on full dataset (using best lambda_weight from tuning)
            logger.info("\nTraining final model with best hyperparameters...")
            trained_model, label_encoder = train_mlp_model(
                embeddings_path=embeddings_path,
                mapping_path=mapping_path,
                dataset_type=dataset_type,
                hidden_dims=best_params['hidden_dims'],
                epochs=best_params['epochs'],
                batch_size=256,  # Use default
                learning_rate=best_params['learning_rate'],
                device=device,
                use_cost_loss=best_params['use_cost_loss'],
                lambda_weight=best_params['lambda_weight'],
                dropout=0.2,  # Use default
                return_val_metrics=False
            )
    else:
        trained_model, label_encoder = train_mlp_model(
            embeddings_path=embeddings_path,
            mapping_path=mapping_path,
            dataset_type=dataset_type,
            hidden_dims=args.hidden_dims,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            use_cost_loss=args.use_cost_loss,
            lambda_weight=args.lambda_weight,
            dropout=args.dropout
        )
    
    if trained_model and label_encoder:
        # Save the model and label encoder
        model_save_path = f'mlp_classifier_{args.dataset}.pth'
        encoder_save_path = f'mlp_label_encoder_{args.dataset}.pkl'
        
        # Get hidden_dims from args or best_params
        if args.use_hyperparameter_tuning:
            hidden_dims = best_params['hidden_dims']
            dropout = 0.2  # Use default
        else:
            hidden_dims = args.hidden_dims
            dropout = args.dropout
        
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'input_dim': trained_model.network[0].in_features,
            'num_classes': trained_model.network[-1].out_features,
        }, model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        joblib.dump(label_encoder, encoder_save_path)
        logger.info(f"Label encoder saved to {encoder_save_path}")
        print("\nModel and label encoder saved successfully.")
    else:
        logger.error("Model training failed or returned None. Model and encoder not saved.")
        print("\nModel training failed. Check logs for details. Model and encoder not saved.")
