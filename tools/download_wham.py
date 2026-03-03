import urllib.request
import os

weights_dir = 'third_party/wham/checkpoints'
os.makedirs(weights_dir, exist_ok=True)
url = 'https://huggingface.co/yohanshin/WHAM/resolve/main/wham_vitpose_w32.pth'
output_path = os.path.join(weights_dir, 'wham_vitpose_w32.pth')

print(f"Downloading WHAM weights from {url}...")
urllib.request.urlretrieve(url, output_path)
print(f"Saved {os.path.getsize(output_path)} bytes to {output_path}")
