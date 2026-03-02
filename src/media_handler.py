import os
import yt_dlp
import uuid
import requests
from urllib.parse import urlparse
from settings import TEMP_DIR

class VideoFetcher:
    """
    Handles ingesting video from local paths, direct URLs, or YouTube links.
    Downloads remote videos to a temporary directory.
    """
    def __init__(self, temp_dir=TEMP_DIR):
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def is_url(self, path: str) -> bool:
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def is_youtube_url(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        return 'youtube.com' in domain or 'youtu.be' in domain

    def download_youtube(self, url: str) -> str:
        unique_id = str(uuid.uuid4())[:8]
        output_template = os.path.join(self.temp_dir, f"yt_{unique_id}.%(ext)s")
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': True,
        }
        
        print(f"📥 Downloading YouTube video from {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', None)
            video_ext = info_dict.get('ext', 'mp4')
            final_path = os.path.join(self.temp_dir, f"yt_{unique_id}.{video_ext}")
            print(f"✅ Downloaded: {video_title} to {final_path}")
            return final_path

    def download_direct_url(self, url: str) -> str:
        unique_id = str(uuid.uuid4())[:8]
        final_path = os.path.join(self.temp_dir, f"vid_{unique_id}.mp4")
        
        print(f"📥 Downloading direct video URL: {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(final_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"✅ Downloaded direct video to {final_path}")
        return final_path

    def get_video_path(self, input_source: str) -> str:
        """
        Takes an input source and returns a valid local file path.
        """
        if not self.is_url(input_source):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Local video not found: {input_source}")
            return input_source
            
        if self.is_youtube_url(input_source):
            return self.download_youtube(input_source)
        else:
            return self.download_direct_url(input_source)
