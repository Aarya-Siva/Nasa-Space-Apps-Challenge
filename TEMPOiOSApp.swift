//
//  TEMPOAIModel.swift
//  TEMPO Air Quality App
//
//  Created for NASA Apps Challenge
//  Date: 2024
//

import Foundation
import CoreLocation
import SwiftUI

// MARK: - Data Models

struct TEMPOPrediction: Codable {
    let no2Value: Double
    let airQuality: String
    let airQualityText: String
    let color: String
    let latitude: Double
    let longitude: Double
    let timestamp: String
    
    enum CodingKeys: String, CodingKey {
        case no2Value = "no2_value"
        case airQuality = "air_quality"
        case airQualityText = "air_quality_text"
        case color
        case latitude
        case longitude
        case timestamp
    }
}

struct TEMPOAPIResponse: Codable {
    let success: Bool
    let prediction: TEMPOPrediction?
    let predictions: [TEMPOPrediction]?
    let count: Int?
    let error: String?
}

struct TEMPOLocation: Codable {
    let latitude: Double
    let longitude: Double
    let name: String
}

struct TEMPORequest: Codable {
    let latitude: Double
    let longitude: Double
    let month: Int?
    let dayOfYear: Int?
    let isWeekend: Int?
    let temperature: Double?
    let windSpeed: Double?
    let humidity: Double?
    
    enum CodingKeys: String, CodingKey {
        case latitude, longitude, month
        case dayOfYear = "day_of_year"
        case isWeekend = "is_weekend"
        case temperature
        case windSpeed = "wind_speed"
        case humidity
    }
}

struct TEMPOBatchRequest: Codable {
    let locations: [TEMPOLocation]
    let month: Int?
    let dayOfYear: Int?
    let isWeekend: Int?
    let temperature: Double?
    let windSpeed: Double?
    let humidity: Double?
    
    enum CodingKeys: String, CodingKey {
        case locations, month
        case dayOfYear = "day_of_year"
        case isWeekend = "is_weekend"
        case temperature
        case windSpeed = "wind_speed"
        case humidity
    }
}

// MARK: - Air Quality Levels

enum AirQualityLevel: String, CaseIterable {
    case cleanAir = "CLEAN_AIR"
    case lowPollution = "LOW_POLLUTION"
    case moderatePollution = "MODERATE_POLLUTION"
    case highPollution = "HIGH_POLLUTION"
    
    var displayName: String {
        switch self {
        case .cleanAir: return "Clean Air"
        case .lowPollution: return "Low Pollution"
        case .moderatePollution: return "Moderate Pollution"
        case .highPollution: return "High Pollution"
        }
    }
    
    var color: Color {
        switch self {
        case .cleanAir: return .green
        case .lowPollution: return .yellow
        case .moderatePollution: return .orange
        case .highPollution: return .red
        }
    }
    
    var emoji: String {
        switch self {
        case .cleanAir: return "✅"
        case .lowPollution: return "🟢"
        case .moderatePollution: return "🟡"
        case .highPollution: return "🔴"
        }
    }
}

// MARK: - TEMPO AI Service

class TEMPOAIService: ObservableObject {
    static let shared = TEMPOAIService()
    
    private let baseURL = "http://localhost:8080" // Change to your server URL
    private let session = URLSession.shared
    
    @Published var isConnected = false
    @Published var lastPrediction: TEMPOPrediction?
    @Published var errorMessage: String?
    
    private init() {
        checkConnection()
    }
    
    // MARK: - Connection Check
    
    func checkConnection() {
        guard let url = URL(string: "\(baseURL)/health") else {
            isConnected = false
            return
        }
        
        session.dataTask(with: url) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.isConnected = false
                    self?.errorMessage = error.localizedDescription
                } else if let data = data,
                          let healthResponse = try? JSONDecoder().decode([String: Any].self, from: data) {
                    self?.isConnected = healthResponse["status"] as? String == "healthy"
                } else {
                    self?.isConnected = false
                }
            }
        }.resume()
    }
    
    // MARK: - Single Prediction
    
    func predictNO2(latitude: Double, longitude: Double, 
                   month: Int? = nil, dayOfYear: Int? = nil,
                   isWeekend: Int? = nil, temperature: Double? = nil,
                   windSpeed: Double? = nil, humidity: Double? = nil,
                   completion: @escaping (Result<TEMPOPrediction, Error>) -> Void) {
        
        let request = TEMPORequest(
            latitude: latitude,
            longitude: longitude,
            month: month,
            dayOfYear: dayOfYear,
            isWeekend: isWeekend,
            temperature: temperature,
            windSpeed: windSpeed,
            humidity: humidity
        )
        
        makeRequest(endpoint: "/predict", request: request) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let response):
                    if response.success, let prediction = response.prediction {
                        self?.lastPrediction = prediction
                        completion(.success(prediction))
                    } else {
                        let error = NSError(domain: "TEMPOAI", code: 1, 
                                          userInfo: [NSLocalizedDescriptionKey: response.error ?? "Unknown error"])
                        completion(.failure(error))
                    }
                case .failure(let error):
                    completion(.failure(error))
                }
            }
        }
    }
    
    // MARK: - Batch Predictions
    
    func predictNO2Batch(locations: [TEMPOLocation],
                        month: Int? = nil, dayOfYear: Int? = nil,
                        isWeekend: Int? = nil, temperature: Double? = nil,
                        windSpeed: Double? = nil, humidity: Double? = nil,
                        completion: @escaping (Result<[TEMPOPrediction], Error>) -> Void) {
        
        let request = TEMPOBatchRequest(
            locations: locations,
            month: month,
            dayOfYear: dayOfYear,
            isWeekend: isWeekend,
            temperature: temperature,
            windSpeed: windSpeed,
            humidity: humidity
        )
        
        makeRequest(endpoint: "/predict_batch", request: request) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let response):
                    if response.success, let predictions = response.predictions {
                        completion(.success(predictions))
                    } else {
                        let error = NSError(domain: "TEMPOAI", code: 1,
                                          userInfo: [NSLocalizedDescriptionKey: response.error ?? "Unknown error"])
                        completion(.failure(error))
                    }
                case .failure(let error):
                    completion(.failure(error))
                }
            }
        }
    }
    
    // MARK: - Current Location Prediction
    
    func predictForCurrentLocation(completion: @escaping (Result<TEMPOPrediction, Error>) -> Void) {
        LocationManager.shared.getCurrentLocation { [weak self] result in
            switch result {
            case .success(let location):
                self?.predictNO2(latitude: location.coordinate.latitude,
                               longitude: location.coordinate.longitude,
                               completion: completion)
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func makeRequest<T: Codable>(endpoint: String, request: T,
                                       completion: @escaping (Result<TEMPOAPIResponse, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)\(endpoint)") else {
            completion(.failure(NSError(domain: "TEMPOAI", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(request)
        } catch {
            completion(.failure(error))
            return
        }
        
        session.dataTask(with: urlRequest) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "TEMPOAI", code: 0, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            do {
                let response = try JSONDecoder().decode(TEMPOAPIResponse.self, from: data)
                completion(.success(response))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
}

// MARK: - Location Manager

class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    static let shared = LocationManager()
    
    private let locationManager = CLLocationManager()
    @Published var currentLocation: CLLocation?
    @Published var authorizationStatus: CLAuthorizationStatus = .notDetermined
    
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }
    
    func requestLocationPermission() {
        locationManager.requestWhenInUseAuthorization()
    }
    
    func getCurrentLocation(completion: @escaping (Result<CLLocation, Error>) -> Void) {
        if authorizationStatus == .authorizedWhenInUse || authorizationStatus == .authorizedAlways {
            if let location = currentLocation {
                completion(.success(location))
            } else {
                locationManager.requestLocation()
                // Store completion for later use
                self.locationCompletion = completion
            }
        } else {
            completion(.failure(NSError(domain: "LocationManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Location permission not granted"])))
        }
    }
    
    private var locationCompletion: ((Result<CLLocation, Error>) -> Void)?
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        currentLocation = location
        if let completion = locationCompletion {
            completion(.success(location))
            locationCompletion = nil
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        if let completion = locationCompletion {
            completion(.failure(error))
            locationCompletion = nil
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        authorizationStatus = status
    }
}

// MARK: - SwiftUI Views

struct TEMPOPredictionView: View {
    @StateObject private var tempoService = TEMPOAIService.shared
    @StateObject private var locationManager = LocationManager.shared
    @State private var prediction: TEMPOPrediction?
    @State private var isLoading = false
    @State private var errorMessage: String?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Connection Status
                HStack {
                    Circle()
                        .fill(tempoService.isConnected ? Color.green : Color.red)
                        .frame(width: 12, height: 12)
                    Text(tempoService.isConnected ? "Connected to TEMPO AI" : "Disconnected")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // Prediction Display
                if let prediction = prediction {
                    VStack(spacing: 15) {
                        Text("Air Quality Prediction")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        VStack(spacing: 10) {
                            Text(AirQualityLevel(rawValue: prediction.airQuality)?.emoji ?? "🌫️")
                                .font(.system(size: 60))
                            
                            Text(AirQualityLevel(rawValue: prediction.airQuality)?.displayName ?? prediction.airQualityText)
                                .font(.title)
                                .fontWeight(.semibold)
                                .foregroundColor(AirQualityLevel(rawValue: prediction.airQuality)?.color ?? .primary)
                            
                            Text("NO₂: \(String(format: "%.2e", prediction.no2Value)) molecules/cm²")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                        
                        Text("Location: \(String(format: "%.3f", prediction.latitude)), \(String(format: "%.3f", prediction.longitude))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } else if isLoading {
                    VStack {
                        ProgressView()
                            .scaleEffect(1.5)
                        Text("Analyzing air quality...")
                            .font(.headline)
                            .padding(.top)
                    }
                } else {
                    VStack {
                        Image(systemName: "location.circle")
                            .font(.system(size: 60))
                            .foregroundColor(.blue)
                        Text("Tap to get air quality prediction")
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                }
                
                // Action Buttons
                VStack(spacing: 12) {
                    Button(action: predictForCurrentLocation) {
                        HStack {
                            Image(systemName: "location.fill")
                            Text("Predict for Current Location")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(!tempoService.isConnected || isLoading)
                    
                    Button(action: predictForNYC) {
                        HStack {
                            Image(systemName: "building.2.fill")
                            Text("Predict for New York City")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(!tempoService.isConnected || isLoading)
                }
                
                if let errorMessage = errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("TEMPO Air Quality")
            .onAppear {
                locationManager.requestLocationPermission()
            }
        }
    }
    
    private func predictForCurrentLocation() {
        isLoading = true
        errorMessage = nil
        
        tempoService.predictForCurrentLocation { result in
            isLoading = false
            switch result {
            case .success(let prediction):
                self.prediction = prediction
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }
    
    private func predictForNYC() {
        isLoading = true
        errorMessage = nil
        
        tempoService.predictNO2(latitude: 40.7128, longitude: -74.0060) { result in
            isLoading = false
            switch result {
            case .success(let prediction):
                self.prediction = prediction
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }
}

// MARK: - App Entry Point

@main
struct TEMPOApp: App {
    var body: some Scene {
        WindowGroup {
            TEMPOPredictionView()
        }
    }
}
