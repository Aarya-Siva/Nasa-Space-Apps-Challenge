#!/usr/bin/env python3
"""
Quick TEMPO AI Model Test

Simple and fast testing script for immediate validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def quick_test():
    print("🚀 QUICK TEMPO AI MODEL TEST")
    print("="*40)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        'latitude': np.random.uniform(25, 50, n_samples),
        'longitude': np.random.uniform(-125, -65, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day_of_year': np.random.randint(1, 366, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'temperature': np.random.uniform(0, 40, n_samples),
        'wind_speed': np.random.uniform(0, 20, n_samples),
        'humidity': np.random.uniform(20, 80, n_samples),
    })
    
    # Create realistic NO2 values
    data['NO2'] = (1e15 + 
        0.5e15 * np.exp(-((data['latitude'] - 40)**2 + (data['longitude'] + 100)**2) / 100) +
        0.3e15 * np.sin(data['day_of_year'] * 2 * np.pi / 365) +
        0.2e15 * (1 - data['is_weekend']) +
        0.1e15 * np.random.random(n_samples))
    
    print(f"✓ Generated {len(data):,} test samples")
    
    # Prepare data
    features = ['latitude', 'longitude', 'month', 'day_of_year', 'is_weekend', 'temperature', 'wind_speed', 'humidity']
    X = data[features]
    y = data['NO2']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🤖 Training model...")
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Test model
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  • Accuracy: {r2*100:.1f}%")
    print(f"  • R² Score: {r2:.3f}")
    print(f"  • RMSE: {rmse:.2e}")
    
    # Test specific predictions
    print(f"\n🎯 SAMPLE PREDICTIONS:")
    
    test_cases = [
        ("NYC", 40.7, -74.0, 8, 0),
        ("LA", 34.0, -118.2, 8, 0),
        ("Rural", 39.0, -98.0, 8, 0),
    ]
    
    for name, lat, lon, month, weekend in test_cases:
        test_data = pd.DataFrame({
            'latitude': [lat],
            'longitude': [lon],
            'month': [month],
            'day_of_year': [month * 30],
            'is_weekend': [weekend],
            'temperature': [25],
            'wind_speed': [10],
            'humidity': [60],
        })
        
        test_scaled = scaler.transform(test_data[features])
        prediction = model.predict(test_scaled)[0]
        
        print(f"  • {name:6s}: {prediction:.2e} molecules/cm²")
    
    print(f"\n✅ TEST COMPLETE!")
    print(f"🎯 Model Status: {'PASS' if r2 > 0.8 else 'FAIL'}")
    
    return r2 > 0.8

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("🎉 Model is ready for production!")
    else:
        print("⚠️  Model needs improvement")
