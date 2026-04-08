#!/usr/bin/env python3
"""
Test script for Mental Health AI System
This script tests the core functionality without running the web server
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def test_dataset_loading():
    """Test if the dataset can be loaded correctly"""
    print("🔍 Testing dataset loading...")
    try:
        df = pd.read_csv('Mental_health_dset.csv')
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Target classes: {df['Disorder'].unique()}")
        print(f"   - Sample count per class:")
        for disorder in df['Disorder'].unique():
            count = len(df[df['Disorder'] == disorder])
            print(f"     • {disorder}: {count}")
        return True
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

def test_model_training():
    """Test if the ML model can be trained"""
    print("\n🤖 Testing model training...")
    try:
        # Load dataset
        df = pd.read_csv('Mental_health_dset.csv')
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != 'Disorder']
        X = df[feature_columns]
        y = df['Disorder']
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Encode features (convert yes/no to 1/0)
        X_encoded = X.apply(lambda x: (x == 'yes').astype(int))
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_encoded, y_encoded)
        
        # Test prediction
        sample_features = X_encoded.iloc[0:1]
        prediction = model.predict(sample_features)[0]
        disorder = label_encoder.inverse_transform([prediction])[0]
        
        print(f"✅ Model trained successfully!")
        print(f"   - Features: {len(feature_columns)}")
        print(f"   - Training samples: {len(X_encoded)}")
        print(f"   - Sample prediction: {disorder}")
        
        # Save model and encoder
        joblib.dump(model, 'mental_health_model.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        print(f"✅ Model and encoder saved to disk")
        
        return True
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        return False

def test_model_prediction():
    """Test if the trained model can make predictions"""
    print("\n🔮 Testing model predictions...")
    try:
        # Load saved model and encoder
        model = joblib.load('mental_health_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Create sample test data
        sample_data = {
            'feeling.nervous': 'yes',
            'panic': 'no',
            'breathing.rapidly': 'yes',
            'having.trouble.in.sleeping': 'yes',
            'having.trouble.with.work': 'no',
            'hopelessness': 'no',
            'anger': 'yes',
            'over.react': 'no',
            'weight.gain': 'no',
            'material.possessions': 'no',
            'introvert': 'yes',
            'popping.up.stressful.memory': 'yes',
            'having.nightmares': 'no',
            'avoids.people.or.activities': 'yes',
            'feeling.negative': 'yes',
            'trouble.concentrating': 'yes',
            'blamming.yourself': 'no'
        }
        
        # Get feature names from the model
        df = pd.read_csv('Mental_health_dset.csv')
        feature_columns = [col for col in df.columns if col != 'Disorder']
        
        # Convert sample data to feature vector
        features = []
        for feature in feature_columns:
            if feature in sample_data:
                features.append(1 if sample_data[feature] == 'yes' else 0)
            else:
                features.append(0)
        
        # Make prediction
        prediction = model.predict([features])[0]
        disorder = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores
        confidence_scores = model.predict_proba([features])[0]
        max_confidence = max(confidence_scores)
        severity_scale = int(max_confidence * 5) + 1
        
        print(f"✅ Prediction successful!")
        print(f"   - Predicted disorder: {disorder}")
        print(f"   - Severity scale: {severity_scale}/5")
        print(f"   - Confidence: {max_confidence:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to make prediction: {e}")
        return False

def test_file_structure():
    """Test if all required files are present"""
    print("\n📁 Testing file structure...")
    required_files = [
        'Mental_health_dset.csv',
        'app.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = ['templates']
    
    template_files = [
        'templates/index.html',
        'templates/chatbot.html',
        'templates/report.html'
    ]
    
    all_good = True
    
    # Check required files
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_good = False
    
    # Check required directories
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - MISSING")
            all_good = False
    
    # Check template files
    for template in template_files:
        if os.path.exists(template):
            print(f"✅ {template}")
        else:
            print(f"❌ {template} - MISSING")
            all_good = False
    
    return all_good

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    test_files = ['mental_health_model.pkl', 'label_encoder.pkl']
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Removed {file}")
        else:
            print(f"ℹ️  {file} not found")

def main():
    """Run all tests"""
    print("🚀 Mental Health AI System - System Test")
    print("=" * 50)
    
    # Test file structure
    if not test_file_structure():
        print("\n❌ File structure test failed. Please check your project setup.")
        return
    
    # Test dataset loading
    if not test_dataset_loading():
        print("\n❌ Dataset loading test failed. Please check your dataset file.")
        return
    
    # Test model training
    if not test_model_training():
        print("\n❌ Model training test failed. Please check your dependencies.")
        return
    
    # Test model prediction
    if not test_model_prediction():
        print("\n❌ Model prediction test failed. Please check your model files.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your system is ready to run.")
    print("\nTo start the web application, run:")
    print("   python app.py")
    print("\nThen open your browser to: http://localhost:5000")
    
    # Clean up test files
    cleanup_test_files()

if __name__ == "__main__":
    main()

