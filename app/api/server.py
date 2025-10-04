from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

def calculate_cloud_seeding_likelihood(lat, lon):
    """
    Calculate the likelihood of successful cloud seeding based on coordinates.
    This is a simplified mock implementation - in a real scenario, you would:
    1. Check real-time weather data
    2. Analyze cloud formation patterns
    3. Consider temperature and humidity
    4. Check wind patterns
    5. Analyze historical success rates
    """
    # Mock calculation using latitude and time of year
    current_month = datetime.now().month
    
    # Simplified factors:
    # - Distance from equator (latitude) affects cloud formation
    # - Seasonal variations (month) affect precipitation likelihood
    base_likelihood = np.cos(np.abs(lat) * np.pi / 180) * 0.5  # Higher near equator
    seasonal_factor = np.sin(current_month * np.pi / 6) * 0.3   # Seasonal variation
    
    # Add some randomness to simulate other environmental factors
    random_factor = np.random.normal(0, 0.1)
    
    likelihood = base_likelihood + seasonal_factor + random_factor
    
    # Ensure likelihood stays between 0 and 1
    likelihood = max(0, min(1, likelihood))
    
    return likelihood

@app.route('/api/cloud-seeding', methods=['GET'])
def get_cloud_seeding_likelihood():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({
                'error': 'Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180'
            }), 400
            
        likelihood = calculate_cloud_seeding_likelihood(lat, lon)
        
        return jsonify({
            'success': True,
            'data': {
                'latitude': lat,
                'longitude': lon,
                'likelihood': round(likelihood, 3),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except (TypeError, ValueError):
        return jsonify({
            'error': 'Invalid parameters. Please provide valid lat and lon query parameters'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)