import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def remove_corrupted_images(directory):
    """
    Goes through all images in a directory and removes the ones that are corrupted.
    Args:
        directory (str): Path to the directory containing images.
    """
    corrupted_files = 0
    total_files = 0

    print(f"Scanning directory: {directory}")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file
        if not os.path.isfile(file_path):
            continue

        total_files += 1
        try:
            # Attempt to open the image
            with Image.open(file_path) as img:
                img.verify()  # Verify the image is not corrupted
        except (IOError, SyntaxError):
            print(f"Removing corrupted image: {file_path}")
            # os.remove(file_path)
            corrupted_files += 1

    print(f"Total files scanned: {total_files}")
    print(f"Corrupted files removed: {corrupted_files}")

if __name__ == "__main__":
    # Replace with the path to your directory
    image_directory = "/home/anubhab-pg/CHAT-AS-MULTIMODAL/data/PhotoChat/images"
    remove_corrupted_images(image_directory)
