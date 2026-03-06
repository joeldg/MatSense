import os
import subprocess
import uuid
import requests
from urllib.parse import urlparse
from settings import TEMP_DIR

class VideoFetcher:
    """
    Handles ingesting video from local paths, direct URLs, or YouTube links.
    Downloads remote videos to a temporary directory.
    Uses yt-dlp (preferred) or pytubefix as fallback.
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
        final_filename = f"yt_{unique_id}.mp4"
        final_path = os.path.join(self.temp_dir, final_filename)
        
        print(f"📥 Downloading YouTube video from {url}...")
        
        # Try yt-dlp first (most reliable, actively maintained)
        try:
            result = subprocess.run([
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", final_path,
                "--no-playlist",
                "--quiet", "--progress",
                url
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and os.path.exists(final_path):
                print(f"✅ Downloaded (yt-dlp): {final_path}")
                return final_path
            else:
                raise RuntimeError(f"yt-dlp failed: {result.stderr[:200]}")
                
        except FileNotFoundError:
            print("   ⚠️ yt-dlp not found, trying pytubefix fallback...")
            # Fallback to pytubefix
            try:
                from pytubefix import YouTube
                from pytubefix.cli import on_progress
                yt = YouTube(url, client='WEB', on_progress_callback=on_progress)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                if not stream:
                    stream = yt.streams.filter(file_extension='mp4').first()
                if not stream:
                    raise ValueError("No suitable MP4 stream found.")
                stream.download(output_path=self.temp_dir, filename=final_filename)
                print(f"✅ Downloaded (pytubefix): {final_path}")
                return final_path
            except Exception as e:
                raise RuntimeError(f"YouTube download failed (both yt-dlp and pytubefix): {e}")

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
