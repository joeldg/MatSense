import sys
import logging
logging.basicConfig(level=logging.DEBUG)

print("Starting WHAM instantiation test")
sys.path.insert(0, 'third_party/wham')

try:
    print("Loading WHAM_API...")
    from wham_api import WHAM_API
    model = WHAM_API()
    print("WHAM_API loaded.")
except Exception as e:
    print("Error:", e)
