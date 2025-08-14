"""
Code to download and preprocess the DialogCC dataset
Link: https://huggingface.co/datasets/passing2961/dialogcc
Paper: https://arxiv.org/pdf/2212.04119v1.pdf 

Code to download and preprocess the PhotoChat Dataset
Link: https://github.com/google-research/google-research/tree/master/multimodalchat/photochat
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image
import requests
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from functools import partial
from timeit import default_timer as timer

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMAGE_NAMES = os.path.join(DATA_DIR, "image_names")
SPLITS = ['train', 'validation', 'test']

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(image, err_file):
    url, index = image
    try:
        start = timer()
        response = requests.get(url)
        end = timer()
        
        if end - start > 10:
            print(url) 
            return False
        if response.status_code==200:
            image_name = f'{index}_{url.split("/")[-1]}.jpg'
            save_path = os.path.join(IMAGE_DIR, image_name)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        ...
    return False
        
def apply_correction(dialogues):
    corrected_dialogues = []
    for dialogue in tqdm(dialogues, desc='correcting dialogs'):
        corrected_dialogue = []
        for i in range(1, len(dialogue)):
            utterance = dialogue[i]
            # print(utterance)
            prev_utterance = dialogue[i-1]
            if utterance['shared_image'] is not []:
                prev_utterance['shared_image'] = utterance['shared_image']
            if prev_utterance['utterance'] is not "":
                corrected_dialogue.append(prev_utterance)
            if i==len(dialogue)-1 and utterance['shared_image'] is []:
                corrected_dialogue.append(utterance)
        corrected_dialogues.append(corrected_dialogue)
    return corrected_dialogues
            

index = 0
def process_dataset(split):
    global index
    dataset = load_dataset("passing2961/dialogcc", split=split)
    df = pd.DataFrame(dataset)
    df['dialogue'] = df['dialogue'].apply(json.dumps)
    
    df['dialogue'] = apply_correction(df['dialogue'].apply(json.loads))
    
    csv_path = os.path.join(DATA_DIR, f'{split}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved at {csv_path}')
    
    image_urls = set() # image_url -> only unique ones
    for dialogue in tqdm(dataset['dialogue'], desc='extract images'):
        for utterance in dialogue:  
            if 'shared_image' in utterance and utterance['shared_image'] is not []:
                for image in utterance['shared_image']:
                    image_urls.add(image['image_url'])
                    
    images = [] # set of (url, index)
    for url in tqdm(image_urls, desc="index images"):
        images.append((url, index))
        index+=1
    
    for i in range(0, len(images), 5000):
        name_path = os.path.join(IMAGE_NAMES, f'{split}_image_urls_{i}_{i+5000}.txt')
        with open(name_path, 'w+') as f:
            for j in range(i, min(len(images), i+5000), 1):
                f.write(f'{images[j][0]}\t{images[j][1]}.jpg\n')
            
                      
                    
    print(f'Downloading {len(image_urls)} images for {split}...')
    
    
class DialogCC(Dataset):
    def __init__(self, csv, img_dir, transform_config=None):
        self.data = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform_config
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dialog = json.loads(row['dialog'])
        
        text = " <|EOU|> ".join([utterance['utterance'] for utterance in dialog])
        image_info = next((utterance['shared_image'][0] for utterance in dialog if 'shared_image' in utterance), None)
        if image_info:
            image_path = os.path.join(self.img_dir, image_info['image_url'].split('/')[-1])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            image = torch.zeros((3, 224, 224))

        return {
            'text': text,
            'image': image,
            'dialogue_id': row['dialogue_id'],
        }


import os
import json
import pandas as pd

def get_photochat(par_folder, split, output_csv):
    data = []
    print("SPLIT: ", split)
    for file in os.listdir(par_folder):
        if split in os.path.basename(file):
            print(file)
            file_path = os.path.join(par_folder, file) 
            with open(file_path, "r") as f:
                dialogues = json.load(f)
                
            for idx, dialogue in enumerate(dialogues):
                data.append({
                    "dialogue_id": dialogue['dialogue_id'],
                    "dialogue": dialogue, 
                    "split": split,
                    "image_url": dialogue['photo_url'],
                    "image_id": dialogue['photo_id'].split('/')[0]+'_'+dialogue['photo_id'].split('/')[1]
                })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    
def get_mmdd(par_folder, split, output_csv):
    data = []
    print("SPLIT:", split)
    
    for file in os.listdir(par_folder):
        if split in os.path.basename(file):
            print(file)
            file_path = os.path.join(par_folder, file)
            
            with open(file_path, "r") as f:
                dialogues = json.load(f)
                
            for dialogue in dialogues:
                dialogue_id = f"{dialogue['dialog_dataset']}_{dialogue['dialog_file'].split('.')[0]}"
                
                formatted_dialog = []
                for idx, message in enumerate(dialogue['dialog']):
                    formatted_dialog.append({
                        'message': message,
                        'share_photo': idx == dialogue['replaced_idx'],
                        'user_id': idx % 2  # Alternating 0/1
                    })
                
                data.append({
                    "dialogue_id": dialogue_id,
                    "dialogue": json.dumps({"dialogue": formatted_dialog}),  # Store as JSON string
                    "split": split,
                    "image_id": dialogue['img_file']
                })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")

# Example usage
            
if __name__ == '__main__':
    # PH_SPLITS=['train', 'test', 'dev']
    # for split in PH_SPLITS:
    #     get_photochat('../../data/PhotoChat/', split, f'../../data/PhotoChat/{split}.csv')  
        
    MMDD_SPLITS=['train', 'test', 'dev']
    for split in MMDD_SPLITS:
        get_mmdd('../../data/MMDD/', split, f'../../data/MMDD/{split}.csv')
          
    # for split in SPLITS:
    #     process_dataset(split)
    
    # train_dataset = DialogCC(os.path.join(DATA_DIR, "train.csv"), IMAGE_DIR)
    # test_dataset = DialogCC(os.path.join(DATA_DIR, "test.csv"), IMAGE_DIR)
    # val_dataset = DialogCC(os.path.join(DATA_DIR, "validation.csv"), IMAGE_DIR)
    
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(val_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")
    
    # train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=32, shuffle=True)
    
    # for batch in train_loader:
    #     print(f"Batch size: {batch['text'].size(0)}")
    #     print(f"Image shape: {batch['image'].shape}")
    #     print(f"Sample text: {batch['text'][0][:100]}...")  # Print first 100 chars of first text in batch
    #     break
    