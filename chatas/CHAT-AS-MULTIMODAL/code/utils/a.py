import requests

url = "https://c3.staticflickr.com/6/5250/5273985737_c0f0e3c247_o.jpg"
save_path = "downloaded_image.jpg"

def download_image(url, save_path):
    try:
        # Custom headers to simulate a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Referer": "https://www.flickr.com/",  # Simulate coming from the Flickr website
        }
        
        # Make the GET request with headers
        response = requests.get(url, headers=headers, timeout=(5, 10))
        
        if response.status_code == 200:
            # Save the image to the specified path
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Image downloaded successfully and saved as '{save_path}'")
        else:
            print(f"Failed to download image. HTTP Status: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Call the function to download the image
download_image(url, save_path)
