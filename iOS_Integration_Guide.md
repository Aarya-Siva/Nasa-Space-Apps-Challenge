# 🍎 TEMPO AI Model - Swift iOS Integration Guide

Complete guide to integrate your TEMPO AI model into a Swift iOS app.

## 📋 **What We've Built:**

✅ **Python API Server** (`tempo_api_server.py`) - REST API for your AI model  
✅ **Swift iOS App** (`TEMPOiOSApp.swift`) - Complete iOS app with UI  
✅ **Real-time Predictions** - Get NO₂ predictions for any location  
✅ **Air Quality Visualization** - Color-coded air quality levels  

---

## 🚀 **Quick Start (5 minutes):**

### **Step 1: Start the API Server**
```bash
cd "/Users/Aarya/Desktop/Nasa Apps Challenge"
python3 tempo_api_server.py
```

**You'll see:**
```
🚀 Starting TEMPO AI Model API Server...
📱 Ready for iOS app integration!
🌐 API will be available at: http://localhost:8080
```

### **Step 2: Test the API**
Open another terminal and test:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 40.7128, "longitude": -74.0060}'
```

**Expected response:**
```json
{
  "success": true,
  "prediction": {
    "no2_value": 1.06e+15,
    "air_quality": "LOW_POLLUTION",
    "air_quality_text": "Low Pollution",
    "color": "#44AA44",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timestamp": "2024-10-04T12:00:00"
  }
}
```

### **Step 3: Integrate with Your Swift App**

Copy the Swift code from `TEMPOiOSApp.swift` into your Xcode project.

---

## 📱 **Swift Integration Details:**

### **Key Components:**

#### **1. TEMPOAIService Class**
```swift
class TEMPOAIService: ObservableObject {
    func predictNO2(latitude: Double, longitude: Double, completion: @escaping (Result<TEMPOPrediction, Error>) -> Void)
    func predictForCurrentLocation(completion: @escaping (Result<TEMPOPrediction, Error>) -> Void)
    func predictNO2Batch(locations: [TEMPOLocation], completion: @escaping (Result<[TEMPOPrediction], Error>) -> Void)
}
```

#### **2. Data Models**
```swift
struct TEMPOPrediction: Codable {
    let no2Value: Double
    let airQuality: String
    let airQualityText: String
    let color: String
    let latitude: Double
    let longitude: Double
    let timestamp: String
}
```

#### **3. Air Quality Levels**
```swift
enum AirQualityLevel: String, CaseIterable {
    case cleanAir = "CLEAN_AIR"           // ✅ Green
    case lowPollution = "LOW_POLLUTION"    // 🟢 Yellow  
    case moderatePollution = "MODERATE_POLLUTION" // 🟡 Orange
    case highPollution = "HIGH_POLLUTION"  // 🔴 Red
}
```

---

## 🎯 **API Endpoints:**

### **1. Single Prediction**
```http
POST /predict
Content-Type: application/json

{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "month": 8,
  "day_of_year": 220,
  "is_weekend": 0,
  "temperature": 25,
  "wind_speed": 10,
  "humidity": 60
}
```

### **2. Batch Predictions**
```http
POST /predict_batch
Content-Type: application/json

{
  "locations": [
    {"latitude": 40.7128, "longitude": -74.0060, "name": "NYC"},
    {"latitude": 34.0522, "longitude": -118.2437, "name": "LA"}
  ]
}
```

### **3. Health Check**
```http
GET /health
```

### **4. Model Info**
```http
GET /model_info
```

---

## 📱 **iOS App Features:**

### **✅ What Your App Can Do:**

1. **📍 Current Location Prediction**
   - Uses GPS to get user's location
   - Predicts air quality for current position

2. **🏙️ City Predictions**
   - Pre-defined locations (NYC, LA, etc.)
   - Custom coordinate input

3. **📊 Air Quality Visualization**
   - Color-coded indicators
   - Emoji representations
   - NO₂ concentration values

4. **🔄 Real-time Updates**
   - Live connection status
   - Error handling
   - Loading states

---

## 🛠️ **Setup Instructions:**

### **For Development:**

1. **Start API Server:**
   ```bash
   python3 tempo_api_server.py
   ```

2. **Update Base URL in Swift:**
   ```swift
   private let baseURL = "http://localhost:8080" // For simulator
   // For physical device, use your Mac's IP: "http://192.168.1.100:8080"
   ```

3. **Add Permissions to Info.plist:**
   ```xml
   <key>NSLocationWhenInUseUsageDescription</key>
   <string>This app needs location access to predict air quality for your current location.</string>
   ```

### **For Production:**

1. **Deploy API Server:**
   - Use Heroku, AWS, or your preferred cloud service
   - Update baseURL to your production server

2. **Add Error Handling:**
   - Network connectivity checks
   - Offline mode support
   - User-friendly error messages

---

## 🎨 **Customization Options:**

### **1. Add More Cities:**
```swift
let majorCities = [
    TEMPOLocation(latitude: 40.7128, longitude: -74.0060, name: "New York"),
    TEMPOLocation(latitude: 34.0522, longitude: -118.2437, name: "Los Angeles"),
    TEMPOLocation(latitude: 41.8781, longitude: -87.6298, name: "Chicago"),
    // Add more...
]
```

### **2. Custom UI Themes:**
```swift
extension AirQualityLevel {
    var backgroundColor: Color {
        switch self {
        case .cleanAir: return Color.green.opacity(0.1)
        case .lowPollution: return Color.yellow.opacity(0.1)
        case .moderatePollution: return Color.orange.opacity(0.1)
        case .highPollution: return Color.red.opacity(0.1)
        }
    }
}
```

### **3. Add Weather Integration:**
```swift
// Get real weather data and pass to API
let weatherData = WeatherService.shared.getCurrentWeather()
tempoService.predictNO2(
    latitude: location.latitude,
    longitude: location.longitude,
    temperature: weatherData.temperature,
    windSpeed: weatherData.windSpeed,
    humidity: weatherData.humidity
)
```

---

## 🚀 **Next Steps:**

1. **✅ Start the API server**
2. **✅ Copy Swift code to your Xcode project**
3. **✅ Test with simulator**
4. **✅ Deploy to physical device**
5. **🎯 Add your own features!**

---

## 📞 **API Response Examples:**

### **Success Response:**
```json
{
  "success": true,
  "prediction": {
    "no2_value": 1.06e+15,
    "air_quality": "LOW_POLLUTION",
    "air_quality_text": "Low Pollution",
    "color": "#44AA44",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timestamp": "2024-10-04T12:00:00"
  }
}
```

### **Error Response:**
```json
{
  "success": false,
  "error": "latitude must be between 25 and 50"
}
```

---

## 🎯 **Your iOS App is Ready!**

**Features:**
- ✅ Real-time air quality predictions
- ✅ GPS location integration  
- ✅ Beautiful UI with color coding
- ✅ Error handling and loading states
- ✅ Production-ready architecture

**Start the API server and integrate with your Swift app!** 🚀
