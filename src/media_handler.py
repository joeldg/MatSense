import os
from pytubefix import YouTube
from pytubefix.cli import on_progress
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
        
        print(f"📥 Downloading YouTube video from {url}...")
        try:
            # Use WEB client to bypass the po_token manual prompt
            yt = YouTube(url, client='WEB', on_progress_callback=on_progress)
            
            # Auto-fetch title
            video_title = yt.title
            
            # Get highest resolution mp4 stream
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not stream:
                stream = yt.streams.filter(file_extension='mp4').first()
                
            if not stream:
                raise ValueError("No suitable MP4 stream found for this video.")
            
            final_filename = f"yt_{unique_id}.mp4"
            final_path = os.path.join(self.temp_dir, final_filename)
            
            stream.download(output_path=self.temp_dir, filename=final_filename)
            
            print(f"✅ Downloaded: {video_title} to {final_path}")
            return final_path
            
        except Exception as e:
            raise RuntimeError(f"PyTubeFix Error 403 Bypass Failed: {e}")

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
