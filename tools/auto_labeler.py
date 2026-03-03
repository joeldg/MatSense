import os
import json
import time
import shutil
import google.generativeai as genai

# Ensure you have run: export GEMINI_API_KEY="your_api_key"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT = """
You are an elite Brazilian Jiu-Jitsu and Judo black belt.
Watch this short tightly cropped video tensor of a grappling exchange.
Identify the primary technique or sequence being executed by the offensive player.
Choose ONLY from the following categories:
['double_leg', 'single_leg', 'osoto_gari', 'guard_pull', 'toreando_pass', 'scramble_unknown']

Return ONLY a valid JSON object in this exact format:
{"technique": "category_name", "confidence": 0.95, "reasoning": "brief explanation"}
"""

def label_dataset(input_dir="dataset/raw_clips", output_dir="dataset/train"):
    print("🤖 Booting Zero-Shot VLM Grappling Analyst...")
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".mp4"): continue
        
        video_path = os.path.join(input_dir, filename)
        print(f"\n📡 Uploading {filename} to VLM...")
        
        video_file = genai.upload_file(path=video_path)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        try:
            response = model.generate_content([video_file, PROMPT])
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            technique = result.get('technique', 'scramble_unknown')
            
            print(f"   ✅ Labeled: {technique} (Confidence: {result.get('confidence')})")
            
            # Auto-sort the video into the training class folder
            tech_folder = os.path.join(output_dir, technique)
            os.makedirs(tech_folder, exist_ok=True)
            shutil.move(video_path, os.path.join(tech_folder, filename))
            
        except Exception as e:
            print(f"   ❌ Failed to parse response for {filename}: {e}")
            
        genai.delete_file(video_file.name)

if __name__ == "__main__":
    label_dataset()