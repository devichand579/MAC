#!/usr/bin/env python3
"""
Analyze neural classifier hyperparameter tuning results.
Finds best performing models based on:
1. Least latencies (lowest avg_cost) - per lambda weight
2. Top 10 models based on F1 score - per lambda weight
"""

import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict
import pandas as pd


def load_results(json_path: str) -> Dict[str, Any]:
    """Load results from neural classifier JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def group_by_lambda(all_results: List[Dict]) -> Dict[float, List[Dict]]:
    """Group results by lambda_weight"""
    grouped = defaultdict(list)
    for result in all_results:
        lambda_weight = result.get('params', {}).get('lambda_weight')
        if lambda_weight is not None:
            grouped[lambda_weight].append(result)
    return dict(grouped)


def find_best_by_latency(results: List[Dict], top_n: int = 10) -> List[Dict]:
    """Find models with least latencies (lowest avg_cost)"""
    # Filter out results without avg_cost
    valid_results = [
        r for r in results 
        if r.get('metrics', {}).get('avg_cost') is not None
    ]
    
    # Sort by avg_cost (ascending - lower is better)
    sorted_results = sorted(
        valid_results,
        key=lambda x: x['metrics']['avg_cost']
    )
    
    return sorted_results[:top_n]


def find_top_by_f1(results: List[Dict], top_n: int = 10) -> List[Dict]:
    """Find top N models based on F1 score"""
    # Filter out results without F1 score
    valid_results = [
        r for r in results 
        if r.get('metrics', {}).get('combined_score', {}).get('f1') is not None
    ]
    
    # Sort by F1 score (descending - higher is better)
    sorted_results = sorted(
        valid_results,
        key=lambda x: x['metrics']['combined_score']['f1'],
        reverse=True
    )
    
    return sorted_results[:top_n]


def find_config_in_f1_range_with_least_latency(all_results: List[Dict], f1_min: float, f1_max: float) -> Dict:
    """Find configuration with F1 in specified range and least latency (across all lambda weights)"""
    valid_results = [
        r for r in all_results
        if (r.get('metrics', {}).get('combined_score', {}).get('f1') is not None and
            r.get('metrics', {}).get('avg_cost') is not None)
    ]
    
    # Filter by F1 range
    in_range = [
        r for r in valid_results
        if f1_min <= r['metrics']['combined_score']['f1'] <= f1_max
    ]
    
    if not in_range:
        return None
    
    # Find the one with least latency
    best = min(in_range, key=lambda x: x['metrics']['avg_cost'])
    return best


def find_highest_f1_config(all_results: List[Dict]) -> Dict:
    """Find configuration with highest F1 score (across all lambda weights)"""
    valid_results = [
        r for r in all_results
        if r.get('metrics', {}).get('combined_score', {}).get('f1') is not None
    ]
    
    if not valid_results:
        return None
    
    # Find the one with highest F1
    best = max(valid_results, key=lambda x: x['metrics']['combined_score']['f1'])
    return best


def format_model_info(result: Dict, rank: int) -> Dict[str, Any]:
    """Format model information for display"""
    params = result.get('params', {})
    metrics = result.get('metrics', {})
    combined = metrics.get('combined_score', {})
    
    return {
        'rank': rank,
        'trial': result.get('trial', 'N/A'),
        'lambda_weight': params.get('lambda_weight', 'N/A'),
        'hidden_dims': params.get('hidden_dims', 'N/A'),
        'epochs': params.get('epochs', 'N/A'),
        'learning_rate': params.get('learning_rate', 'N/A'),
        'accuracy': metrics.get('accuracy', 'N/A'),
        'avg_cost': metrics.get('avg_cost', 'N/A'),
        'f1': combined.get('f1', 'N/A'),
        'precision': combined.get('precision', 'N/A'),
        'recall': combined.get('recall', 'N/A'),
        'syntactic_match': combined.get('syntactic_match', 'N/A'),
        'score': result.get('score', 'N/A')
    }


def print_results_table(results: List[Dict], title: str, lambda_weight: float = None):
    """Print results in a formatted table"""
    if lambda_weight is not None:
        print(f"\n{'='*120}")
        print(f"{title} - Lambda Weight: {lambda_weight}")
        print(f"{'='*120}")
    else:
        print(f"\n{'='*120}")
        print(f"{title}")
        print(f"{'='*120}")
    
    # Create DataFrame for better formatting
    formatted_results = [format_model_info(r, i+1) for i, r in enumerate(results)]
    df = pd.DataFrame(formatted_results)
    
    # Print table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(df.to_string(index=False))
    print(f"\nTotal models shown: {len(results)}")


def save_results_to_json(results_by_lambda: Dict[float, Dict[str, List[Dict]]], output_path: str):
    """Save results to JSON file organized by lambda weight"""
    output = {}
    for lambda_w in sorted(results_by_lambda.keys()):
        output[str(lambda_w)] = {
            'best_by_latency': [
                format_model_info(r, i+1) for i, r in enumerate(results_by_lambda[lambda_w]['best_by_latency'])
            ],
            'top_by_f1': [
                format_model_info(r, i+1) for i, r in enumerate(results_by_lambda[lambda_w]['top_by_f1'])
            ]
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze neural classifier hyperparameter tuning results"
    )
    parser.add_argument(
        '--input_json',
        type=str,
        default='neural_hyperparameter_mmdd.json',
        help='Path to neural classifier JSON results file'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='neural_classifier_analysis.json',
        help='Path to save analysis results JSON file'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='Number of top models to show (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input_json}...")
    data = load_results(args.input_json)
    
    # Get all results
    all_results = data.get('all_results', [])
    print(f"Total trials found: {len(all_results)}")
    
    if not all_results:
        print("Error: No results found in JSON file!")
        return
    
    # Group results by lambda_weight
    results_by_lambda = group_by_lambda(all_results)
    print(f"\nFound results for {len(results_by_lambda)} lambda weights: {sorted(results_by_lambda.keys())}")
    
    # Process each lambda weight separately
    results_by_lambda_processed = {}
    
    for lambda_w in sorted(results_by_lambda.keys()):
        lambda_results = results_by_lambda[lambda_w]
        print(f"\n{'='*120}")
        print(f"Processing Lambda Weight: {lambda_w} ({len(lambda_results)} trials)")
        print(f"{'='*120}")
        
        # Find best models by latency for this lambda
        best_latency = find_best_by_latency(lambda_results, top_n=args.top_n)
        
        # Find top models by F1 score for this lambda
        top_f1 = find_top_by_f1(lambda_results, top_n=args.top_n)
        
        # Store results
        results_by_lambda_processed[lambda_w] = {
            'best_by_latency': best_latency,
            'top_by_f1': top_f1
        }
        
        # Print results for this lambda
        print_results_table(best_latency, f"TOP {args.top_n} MODELS BY LEAST LATENCY (Lowest avg_cost)", lambda_w)
        print_results_table(top_f1, f"TOP {args.top_n} MODELS BY F1 SCORE", lambda_w)
        
        # Print summary statistics for this lambda
        print(f"\n{'='*120}")
        print(f"SUMMARY STATISTICS - Lambda Weight: {lambda_w}")
        print(f"{'='*120}")
        
        if best_latency:
            avg_costs = [r['metrics']['avg_cost'] for r in best_latency]
            print(f"\nBest Latency Models:")
            print(f"  Min latency: {min(avg_costs):.6f} seconds")
            print(f"  Max latency: {max(avg_costs):.6f} seconds")
            print(f"  Avg latency: {sum(avg_costs)/len(avg_costs):.6f} seconds")
        
        if top_f1:
            f1_scores = [r['metrics']['combined_score']['f1'] for r in top_f1]
            print(f"\nTop F1 Models:")
            print(f"  Min F1: {min(f1_scores):.6f}")
            print(f"  Max F1: {max(f1_scores):.6f}")
            print(f"  Avg F1: {sum(f1_scores)/len(f1_scores):.6f}")
    
    # Save to JSON
    save_results_to_json(results_by_lambda_processed, args.output_json)
    
    # Print overall summary
    print(f"\n{'='*120}")
    print("OVERALL SUMMARY ACROSS ALL LAMBDA WEIGHTS")
    print(f"{'='*120}")
    
    all_best_latencies = []
    all_top_f1s = []
    for lambda_w in sorted(results_by_lambda_processed.keys()):
        all_best_latencies.extend(results_by_lambda_processed[lambda_w]['best_by_latency'])
        all_top_f1s.extend(results_by_lambda_processed[lambda_w]['top_by_f1'])
    
    if all_best_latencies:
        avg_costs = [r['metrics']['avg_cost'] for r in all_best_latencies]
        print(f"\nBest Latency Models (across all lambdas):")
        print(f"  Min latency: {min(avg_costs):.6f} seconds")
        print(f"  Max latency: {max(avg_costs):.6f} seconds")
        print(f"  Avg latency: {sum(avg_costs)/len(avg_costs):.6f} seconds")
    
    if all_top_f1s:
        f1_scores = [r['metrics']['combined_score']['f1'] for r in all_top_f1s]
        print(f"\nTop F1 Models (across all lambdas):")
        print(f"  Min F1: {min(f1_scores):.6f}")
        print(f"  Max F1: {max(f1_scores):.6f}")
        print(f"  Avg F1: {sum(f1_scores)/len(f1_scores):.6f}")
    
    # Find specific configurations based on dataset type - PER LAMBDA WEIGHT
    print(f"\n{'='*120}")
    print("SPECIFIC CONFIGURATION SEARCHES (PER LAMBDA WEIGHT)")
    print(f"{'='*120}")
    
    # Detect dataset type from filename
    dataset_type = None
    if 'mmdd' in args.input_json.lower():
        dataset_type = 'mmdd'
        f1_min, f1_max = 0.24, 0.25
    elif 'imagechat' in args.input_json.lower() or 'image_chat' in args.input_json.lower():
        dataset_type = 'imagechat'
        f1_min, f1_max = 0.22, 0.23
    else:
        # Try to infer from data
        print("Warning: Could not detect dataset type from filename. Skipping specific searches.")
        dataset_type = None
    
    if dataset_type:
        # Search for each lambda weight separately
        for lambda_w in sorted(results_by_lambda.keys()):
            lambda_results = results_by_lambda[lambda_w]
            
            print(f"\n{'='*80}")
            print(f"LAMBDA WEIGHT: {lambda_w}")
            print(f"{'='*80}")
            
            # Find config with F1 in range and least latency for this lambda weight
            config_in_range = find_config_in_f1_range_with_least_latency(lambda_results, f1_min, f1_max)
            if config_in_range:
                print(f"\nConfiguration with F1 between {f1_min} and {f1_max} and LEAST LATENCY:")
                print(f"  Lambda Weight: {config_in_range.get('params', {}).get('lambda_weight', 'N/A')}")
                print(f"  Trial: {config_in_range.get('trial', 'N/A')}")
                print(f"  F1 Score: {config_in_range['metrics']['combined_score']['f1']:.6f}")
                print(f"  Precision: {config_in_range['metrics']['combined_score']['precision']}")
                print(f"  Recall: {config_in_range['metrics']['combined_score']['recall']}")
                print(f"  Syntactic Match: {config_in_range['metrics']['combined_score']['syntactic_match']}")
                print(f"  Latency (avg_cost): {config_in_range['metrics']['avg_cost']:.6f} seconds")
                print(f"  Accuracy: {config_in_range['metrics']['accuracy']:.6f}")
                print(f"  Parameters:")
                params = config_in_range.get('params', {})
                for key, value in params.items():
                    print(f"    {key}: {value}")
            else:
                print(f"\nNo configuration found with F1 between {f1_min} and {f1_max} for lambda weight {lambda_w}")
            
            # Find configuration with highest F1 for this lambda weight
            highest_f1_config = find_highest_f1_config(lambda_results)
            if highest_f1_config:
                print(f"\nConfiguration with HIGHEST F1 SCORE:")
                print(f"  Lambda Weight: {highest_f1_config.get('params', {}).get('lambda_weight', 'N/A')}")
                print(f"  Trial: {highest_f1_config.get('trial', 'N/A')}")
                print(f"  F1 Score: {highest_f1_config['metrics']['combined_score']['f1']:.6f}")
                print(f"  Precision: {highest_f1_config['metrics']['combined_score']['precision']}")
                print(f"  Recall: {highest_f1_config['metrics']['combined_score']['recall']}")
                print(f"  Syntactic Match: {highest_f1_config['metrics']['combined_score']['syntactic_match']}")
                print(f"  Latency (avg_cost): {highest_f1_config['metrics']['avg_cost']:.6f} seconds")
                print(f"  Accuracy: {highest_f1_config['metrics']['accuracy']:.6f}")
                print(f"  Parameters:")
                params = highest_f1_config.get('params', {})
                for key, value in params.items():
                    print(f"    {key}: {value}")
            else:
                print(f"\nNo configuration found with valid F1 score for lambda weight {lambda_w}")
    
    # Also show overall best across all lambda weights
    print(f"\n{'='*120}")
    print("OVERALL BEST (ACROSS ALL LAMBDA WEIGHTS)")
    print(f"{'='*120}")
    
    if dataset_type:
        # Find config with F1 in range and least latency (across all lambda weights)
        config_in_range = find_config_in_f1_range_with_least_latency(all_results, f1_min, f1_max)
        if config_in_range:
            print(f"\nConfiguration with F1 between {f1_min} and {f1_max} and LEAST LATENCY (across all lambda weights):")
            print(f"  Lambda Weight: {config_in_range.get('params', {}).get('lambda_weight', 'N/A')}")
            print(f"  Trial: {config_in_range.get('trial', 'N/A')}")
            print(f"  F1 Score: {config_in_range['metrics']['combined_score']['f1']:.6f}")
            print(f"  Precision: {config_in_range['metrics']['combined_score']['precision']}")
            print(f"  Recall: {config_in_range['metrics']['combined_score']['recall']}")
            print(f"  Syntactic Match: {config_in_range['metrics']['combined_score']['syntactic_match']}")
            print(f"  Latency (avg_cost): {config_in_range['metrics']['avg_cost']:.6f} seconds")
            print(f"  Accuracy: {config_in_range['metrics']['accuracy']:.6f}")
            print(f"  Parameters:")
            params = config_in_range.get('params', {})
            for key, value in params.items():
                print(f"    {key}: {value}")
        else:
            print(f"\nNo configuration found with F1 between {f1_min} and {f1_max} across all lambda weights")
    
    # Find configuration with highest F1 and its latency (across all lambda weights)
    highest_f1_config = find_highest_f1_config(all_results)
    if highest_f1_config:
        print(f"\nConfiguration with HIGHEST F1 SCORE (across all lambda weights):")
        print(f"  Lambda Weight: {highest_f1_config.get('params', {}).get('lambda_weight', 'N/A')}")
        print(f"  Trial: {highest_f1_config.get('trial', 'N/A')}")
        print(f"  F1 Score: {highest_f1_config['metrics']['combined_score']['f1']:.6f}")
        print(f"  Precision: {highest_f1_config['metrics']['combined_score']['precision']}")
        print(f"  Recall: {highest_f1_config['metrics']['combined_score']['recall']}")
        print(f"  Syntactic Match: {highest_f1_config['metrics']['combined_score']['syntactic_match']}")
        print(f"  Latency (avg_cost): {highest_f1_config['metrics']['avg_cost']:.6f} seconds")
        print(f"  Accuracy: {highest_f1_config['metrics']['accuracy']:.6f}")
        print(f"  Parameters:")
        params = highest_f1_config.get('params', {})
        for key, value in params.items():
            print(f"    {key}: {value}")
    else:
        print("\nNo configuration found with valid F1 score")
    
    print(f"\n{'='*120}")


if __name__ == '__main__':
    main()

