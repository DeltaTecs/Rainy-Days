from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)


@app.route('/api/cloud-seeding', methods=['GET'])
def get_cloud_seeding_score():
    """Return dummy cloud seeding score values over an Alberta grid.

    Query Parameters:
        days (int|str): Ignored (present for future expansion / contract change)

    Response:
        JSON containing a list of grid points (lat, lon) spaced 0.25° apart
    covering Alberta (approx 49N–60N, -120W–-110W) each with a seeding score
    float in [0,1].
    """

    # We intentionally ignore the value of 'days' per instructions, but access it
    # so that clients know the parameter is acknowledged.
    _ = request.args.get('days')  # noqa: F841 (explicitly unused)

    lat_min, lat_max = 49.0, 60.0
    lon_min, lon_max = -120.0, -110.0
    step = 0.25

    # Generate ranges (inclusive of max with a tiny epsilon tolerance)
    lats = np.arange(lat_min, lat_max + 1e-9, step)
    lons = np.arange(lon_min, lon_max + 1e-9, step)

    # Dummy seeding scores: random but reproducible within a single process lifetime.
    # For stable values between re-runs, we could seed with a fixed value; here we let it vary.
    grid = []
    for lat in lats:
        for lon in lons:
            score = float(np.random.random())  # already in [0,1)
            grid.append({
                'latitude': round(float(lat), 3),
                'longitude': round(float(lon), 3),
                'seeding_score': round(score, 4)
            })

    return jsonify({
        'success': True,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'meta': {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'step': step,
            'count': len(grid)
        },
        'data': grid
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)