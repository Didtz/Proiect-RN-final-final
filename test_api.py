#!/usr/bin/env python3
"""
Test script for Plant Identifier API
Tests if the model correctly identifies plants from training data
"""

import requests
import os
from pathlib import Path

# API endpoint
API_URL = "http://localhost:5000/api/predict"

# Path to training data
TRAINING_DATA = Path("training_data")

def test_plant_identification():
    """Test plant identification with images from each plant class"""
    
    print("\n" + "="*60)
    print("PLANT IDENTIFIER API TEST")
    print("="*60)
    
    # Test each plant class
    plant_classes = ['cactus', 'orchid', 'rose', 'sunflower', 'tulip']
    results = []
    
    for plant_class in plant_classes:
        plant_folder = TRAINING_DATA / plant_class
        
        if not plant_folder.exists():
            print(f"\n[SKIP] {plant_class}: folder not found")
            continue
        
        # Get first image from folder
        images = list(plant_folder.glob("*.jpg"))
        if not images:
            images = list(plant_folder.glob("*.png"))
        
        if not images:
            print(f"\n[ERROR] {plant_class}: no images found")
            continue
        
        test_image = images[0]
        print(f"\n[TEST] {plant_class.upper()}")
        print(f"  Image: {test_image.name}")
        
        # Test API
        try:
            with open(test_image, 'rb') as f:
                files = {'file': f}
                response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    identified = data.get('plant', 'Unknown')
                    confidence = data.get('confidence', 0)
                    correct = identified == plant_class
                    
                    status = "PASS" if correct else "FAIL"
                    print(f"  Result: {status}")
                    print(f"  Identified: {identified}")
                    print(f"  Confidence: {confidence*100:.1f}%")
                    print(f"  Scientific Name: {data.get('info', {}).get('scientific_name', 'N/A')}")
                    print(f"  Care Info: {data.get('info', {}).get('watering', 'N/A')}")
                    
                    results.append({
                        'plant': plant_class,
                        'identified': identified,
                        'confidence': confidence,
                        'correct': correct
                    })
                else:
                    print(f"  Result: FAIL")
                    print(f"  Error: {data.get('message', 'Unknown error')}")
                    results.append({
                        'plant': plant_class,
                        'identified': None,
                        'confidence': 0,
                        'correct': False
                    })
            else:
                print(f"  Result: ERROR (HTTP {response.status_code})")
                results.append({
                    'plant': plant_class,
                    'identified': None,
                    'confidence': 0,
                    'correct': False
                })
        except Exception as e:
            print(f"  Result: ERROR")
            print(f"  Exception: {e}")
            results.append({
                'plant': plant_class,
                'identified': None,
                'confidence': 0,
                'correct': False
            })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    for result in results:
        symbol = "✓" if result['correct'] else "✗"
        print(f"{symbol} {result['plant']:12} -> {result['identified'] or 'None':12} ({result['confidence']*100:5.1f}%)")
    
    print("-"*60)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("="*60)
    
    return accuracy >= 0.70

if __name__ == "__main__":
    try:
        success = test_plant_identification()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        exit(2)
