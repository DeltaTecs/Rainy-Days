import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();
const port = process.env.PORT || 3000;
const cloudSeedingApiBaseUrl = process.env.CLOUD_SEEDING_API_BASE_URL || 'http://localhost:5000';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const leafletAssetPath = path.join(__dirname, 'node_modules', 'leaflet', 'dist');
app.use('/vendor/leaflet', express.static(leafletAssetPath));

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/predictions', (_req, res) => {
  res.json({
    message: 'Cloud seeding prediction placeholder',
    regions: [
      {
        // Sample region moved to a central Alberta coordinate so the frontend stays focused there.
        name: 'Central Alberta Sample',
        coordinates: { lat: 53.9333, lng: -116.5765 },
        confidence: 0.42
      }
    ]
  });
});

app.get('/api/cloud-seeding', async (req, res) => {
  try {
    const url = new URL('/api/cloud-seeding', cloudSeedingApiBaseUrl);
    Object.entries(req.query || {}).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach(v => url.searchParams.append(key, v));
      } else if (value !== undefined) {
        url.searchParams.append(key, value);
      }
    });

    const response = await fetch(url);
    if (!response.ok) {
      const text = await response.text();
      console.error('Cloud seeding API error response:', text);
      return res.status(response.status).json({
        message: 'Failed to retrieve cloud seeding scores.',
        status: response.status
      });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Error fetching cloud seeding scores:', error);
    res.status(502).json({
      message: 'Failed to reach cloud seeding API.'
    });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
