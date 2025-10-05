import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();
const port = process.env.PORT || 3000;

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

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
