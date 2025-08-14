import pandas as pd
from collections import Counter

def count_common_images(csv_files):
    """
    Count images that appear in at least 2 CSV files from the given list.
    
    Args:
        csv_files (list): List of paths to CSV files containing image_id column
        
    Returns:
        dict: Contains statistics about common images and their distribution
    """
    # Read all image IDs from each CSV
    file_image_sets = {}
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        file_name = csv_file.split('/')[-1].replace('.csv', '')
        file_image_sets[file_name] = set(df['image_id'].tolist())
    
    # Calculate pairwise overlaps
    pairs = [
        ('train', 'test'),
        ('train', 'dev'),
        ('test', 'dev')
    ]
    
    pairwise_overlaps = {}
    total_common_images = set()  # Use a set to avoid duplicates
    
    for file1, file2 in pairs:
        common = file_image_sets[file1] & file_image_sets[file2]
        pairwise_overlaps[f"{file1}-{file2}"] = len(common)
        total_common_images.update(common)  # Add all common images to the set
    
    # Images present in all three files
    in_all_files = len(set.intersection(*file_image_sets.values()))
    
    # Validation
    print("\nValidation:")
    for img_id in total_common_images:
        appearances = sum(1 for img_set in file_image_sets.values() if img_id in img_set)
        if appearances < 2:
            print(f"Warning: Image {img_id} appears in less than 2 files but was counted!")
    
    return {
        'total_common_images': len(total_common_images),
        'pairwise_overlaps': pairwise_overlaps,
        'in_all_files': in_all_files,
        'file_sizes': {name: len(img_set) for name, img_set in file_image_sets.items()}
    }

# Example usage
if __name__ == "__main__":
    csv_files = [
        '../../data/PhotoChat/train.csv',
        '../../data/PhotoChat/test.csv',
        '../../data/PhotoChat/dev.csv'
    ]
    
    results = count_common_images(csv_files)
    
    print("\nImage Distribution Analysis:")
    print(f"Total images appearing in 2 or more files: {results['total_common_images']}")
    print("\nPairwise overlaps:")
    for pair, count in results['pairwise_overlaps'].items():
        print(f"{pair}: {count} images")
    print(f"\nImages present in all three files: {results['in_all_files']}")
    print("\nFile sizes:")
    for file, size in results['file_sizes'].items():
        print(f"{file}: {size} images")