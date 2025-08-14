# import os
# import requests
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm

# DATA_DIR = "data/PhotoChat"
# IMAGE_DIR = os.path.join(DATA_DIR, "images")
# IMAGE_NAMES = os.path.join(DATA_DIR, "image_names")
# SPLITS = ['train', 'test', 'dev']

# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(IMAGE_DIR, exist_ok=True)

# def download_image(image):
#     url, image_name = image
#     try:
#         session = requests.Session()
#         retries = requests.adapters.Retry(total=1, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
#         session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
#         session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
#         response = session.get(url, timeout=(1, 2))
#         if response.status_code==200:
#             # image_name = f'{index}_{url.split("/")[-1]}.jpg'
#             save_path = os.path.join(IMAGE_DIR, image_name)
#             with open(save_path, 'wb') as f:
#                 f.write(response.content)
#             return True
#     except Exception as e:
#         ...
#     return False

# if __name__ == '__main__':
#     for file in tqdm(os.listdir(IMAGE_NAMES), desc="download images"):
#         print('starting ', file)
#         images = []
#         with open(os.path.join(IMAGE_NAMES, file), 'r') as f:
#             for line in f:
#                 url, save_name = line.strip().split('\t')
#                 if '.jpg' not in save_name:
#                     save_name += '.jpg'
#                 images.append((url, save_name))
#         with Pool(cpu_count()) as p:
#             p.map(download_image, images)
#         print('done ', file)
#         print('number of images now: ', len(os.listdir(IMAGE_DIR)))

import os
import requests
from multiprocessing import Pool, cpu_count, Semaphore
from tqdm import tqdm
import pandas as pd
import mimetypes
import time

DATA_DIR = "../../data/PhotoChat"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_FILES = ['../../data/PhotoChat/dev.csv', '../../data/PhotoChat/test.csv', '../../data/PhotoChat/train.csv']  # List of CSV file paths
# CSV_FILES = ['../../data/PhotoChat/test.csv']

os.makedirs(IMAGE_DIR, exist_ok=True)

cnt_error = 0

def download_image(image):
    global cnt_error
    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Referer": "https://www.flickr.com/",
    }
    url, image_id = image
    
    # if image is present in the folder, skip downloading
    if os.path.exists(os.path.join(IMAGE_DIR, f'{image_id}.jpg')):
        return True
    
    try:
        # Create a requests session with retry logic
        session = requests.Session()
        retries = requests.adapters.Retry(total=1, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
        
        while True:
            response = session.get(url, headers=headers, timeout=(1, 2))  # Adjust timeouts as needed
            if response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get("Retry-After", 1))  # Default to 1 second if not provided
                print(f"Rate limit hit. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            elif response.status_code == 200:
                # Guess the file extension based on content type
                content_type = response.headers['content-type']
                ext = mimetypes.guess_extension(content_type)
                if ext in ['.jpeg', '.jpg', '.png']:
                    save_path = os.path.join(IMAGE_DIR, f'{image_id}{ext}')
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    return True
                else:
                    print(f"Unsupported content type for {image_id}: {content_type}")
            else:
                # count 404/410 issues
                if response.status_code == 404 or response.status_code == 410:
                    cnt_error += 1
            break
    # except requests.exceptions.ConnectTimeout:
    #     print(f"Connection timeout while downloading {image_id}. URL: {url}")
    # except requests.exceptions.ReadTimeout:
    #     print(f"Read timeout while downloading {image_id}. URL: {url}")
    # except requests.exceptions.Timeout:
    #     print(f"General timeout while downloading {image_id}. URL: {url}")
    except Exception as e:
        print(f"Error downloading {image_id}: {e}")
    return False

semaphore = Semaphore(1)

def download_image_with_rate_limit(image):
    with semaphore:  # Ensure only one request at a time
        time.sleep(1)  # Enforce a global delay
        return download_image(image)

if __name__ == '__main__':
    for csv_file in CSV_FILES:
        # Use pandas to read the CSV file
        df = pd.read_csv(csv_file)
        
        # Create a list of (image_url, image_id) tuples
        images = list(zip(df['image_url'], df['image_id']))
        
        print("TOTAL IMAGES TO DOWNLOAD: ", len(images))

        # Use multiprocessing to download images in parallel
        with Pool(cpu_count()) as p:
            results = list(tqdm(p.imap(download_image, images), total=len(images), desc=f"Downloading images from {os.path.basename(csv_file)}"))
        
        print(f"Downloaded {sum(results)} images from {os.path.basename(csv_file)}.")
        print(f'Number of errors: {cnt_error}')


