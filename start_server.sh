#!/bin/bash
# TEMPO AI API Server Startup Script

echo "🚀 Starting TEMPO AI Model API Server on Port 8080..."
echo "📱 Ready for iOS app integration!"
echo ""

# Kill any existing processes on port 8080
echo "🔧 Checking for existing processes on port 8080..."
lsof -ti:8080 | xargs kill -9 2>/dev/null && echo "✅ Cleared port 8080" || echo "✅ Port 8080 is free"

echo ""
echo "🌐 API Server will be available at: http://localhost:8080"
echo "📋 Available endpoints:"
echo "  • POST /predict - Single prediction"
echo "  • POST /predict_batch - Multiple predictions" 
echo "  • GET /health - Health check"
echo "  • GET /model_info - Model information"
echo ""
echo "📱 Update your Swift app baseURL to: http://localhost:8080"
echo ""
echo "🎯 Test with: curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{\"latitude\": 40.7128, \"longitude\": -74.0060}'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 tempo_api_server.py
