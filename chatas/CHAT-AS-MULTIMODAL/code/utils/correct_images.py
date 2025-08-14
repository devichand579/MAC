import os
from tqdm import tqdm
def correct_images(image_path):
    if not image_path.endswith('.jpg'):
        # rename image
        new_image_path = image_path.split('.jpg')[0]+'.jpg'
        os.rename(image_path, new_image_path)
def count_err(image_path):
    if not image_path.endswith('.jpg'):
        return True
    return False
        

if __name__ == '__main__':
    count_error = 0
    for file in tqdm(os.listdir('/home/bishals/Multimodal_Chat_AS/data/images')):
        image_path = os.path.join('/home/bishals/Multimodal_Chat_AS/data/images', file)
        count_error += count_err(image_path)
        correct_images(image_path)
        
    print('Total Images in IMAGE_DIR: ', len(os.listdir('/home/bishals/Multimodal_Chat_AS/data/images')))
    print('Total images not ending in .jpg: ', count_error)
    count_error = 0
    for file in os.listdir('/home/bishals/Multimodal_Chat_AS/data/images'):
        image_path = os.path.join('/home/bishals/Multimodal_Chat_AS/data/images', file)
        count_error += count_err(image_path)
    print('Total images not ending in .jpg after change: ', count_error)
    
        