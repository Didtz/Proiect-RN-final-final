#!/usr/bin/env python
"""Quick API test for flower identification"""
import requests
import time
from pathlib import Path

print("Waiting for Flask to be ready...")
time.sleep(2)

test_cases = [
    ("training_data/rose/rose_010.jpg", "rose"),
    ("training_data/cactus/cactus_015.jpg", "cactus"),
    ("training_data/sunflower/sunflower_005.jpg", "sunflower"),
]

url = 'http://127.0.0.1:5000/api/predict'

for image_path, expected in test_cases:
    try:
        if not Path(image_path).exists():
            print(f"✗ {expected.upper()}: File not found - {image_path}")
            continue
            
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(url, files=files, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                plant = result.get('plant', 'Unknown')
                confidence = result.get('confidence', 0)
                match = "✓" if plant.lower() == expected.lower() else "✗"
                print(f"{match} {expected.upper()}: Predicted '{plant}' ({confidence*100:.1f}%)")
            else:
                print(f"✗ {expected.upper()}: HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ {expected.upper()}: {str(e)}")

print("\nWeb Interface Test:")
print("Open browser to: http://127.0.0.1:5000/identify")
print("Upload a flower image and click 'Identificati Planta'")
