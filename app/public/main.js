import * as L from '/vendor/leaflet/leaflet-src.esm.js';

const statusElement = document.getElementById('map-status');
const mapElement = document.getElementById('map');

const updateStatus = (message, isError = false) => {
  statusElement.textContent = message;
  statusElement.style.color = isError ? '#d14343' : '';
};

const fetchPredictions = async () => {
  const response = await fetch('/api/predictions');
  if (!response.ok) {
    throw new Error('Failed to load prediction data.');
  }

  return response.json();
};

const initializeMap = () => {
  const map = L.map(mapElement, {
    // Start roughly centered; we'll refine to exact province bounds once GeoJSON loads.
    center: [53.9333, -116.5765],
    zoom: 5,
    zoomControl: false
  });

  L.control.zoom({ position: 'bottomright' }).addTo(map);

  // Grey out everything except Alberta, Canada.
  fetch('https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/canada.geojson')
    .then(r => r.json())
    .then(provincesData => {
      const albertaFeature = provincesData.features.find(f => /alberta/i.test(f.properties.name));
      const otherProvinces = provincesData.features.filter(f => !/alberta/i.test(f.properties.name));

      // Add other provinces with outline only (no opaque fill) so Canada doesn't look darker than the rest of the world.
      L.geoJSON({ type: 'FeatureCollection', features: otherProvinces }, {
        style: {
          color: '#777',
          weight: 1,
          fillColor: '#000', // kept for compatibility; fully transparent via fillOpacity
          fillOpacity: 0
        }
      }).addTo(map);

      // Highlight Alberta outline.
      if (albertaFeature) {
        const albertaLayer = L.geoJSON(albertaFeature, {
          style: {
            color: '#222',
            weight: 2.5,
            fillColor: '#ffd54f',
            fillOpacity: 0.1
          }
        }).addTo(map);
        map.albertaGeoJSON = albertaFeature; // For click detection

        // Fit to Alberta bounds exactly once on initial load.
        if (!map._albertaFitted) {
          const bounds = albertaLayer.getBounds();
            // Add slight padding so border isn't flush with viewport.
          map.fitBounds(bounds, { padding: [25, 25] });
          map._albertaFitted = true;
        }
      }
    })
    .catch(err => {
      console.error('Failed to render Alberta-focused layering:', err);
    });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  // Removed automatic user geolocation recentering per request.
  updateStatus('Map centered on Alberta, Canada.');

  // Add click event listener to print coordinates and get cloud seeding likelihood
  map.on('click', async (event) => {
    const { lat, lng } = event.latlng;
    
    // Check if the clicked point is within Alberta
    if (map.albertaGeoJSON) {
      const point = {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [lng, lat]
        }
      };

      const isInAlberta = leafletPip.pointInLayer(
        [lng, lat],
        L.geoJSON(map.albertaGeoJSON)
      ).length > 0;

      if (!isInAlberta) {
        console.log('Click ignored - location outside Alberta');
        return;
      }

      console.log(`Clicked location in Alberta - Latitude: ${lat.toFixed(6)}, Longitude: ${lng.toFixed(6)}`);
      
      try {
        const response = await fetch(`http://localhost:5000/api/cloud-seeding?lat=${lat}&lon=${lng}`);
        if (!response.ok) {
          throw new Error('Failed to fetch cloud seeding likelihood');
        }
        const result = await response.json();
        console.log('Cloud Seeding Analysis:', result.data);
      } catch (error) {
        console.error('Error fetching cloud seeding likelihood:', error);
      }
    }
  });

  return map;
};

const ALBERTA_CENTER = L.latLng(53.9333, -116.5765);
const ALBERTA_BOUNDS_PADDING_DEG = 6; // Rough padding box to keep focus in province
const isWithinAlbertaBox = (latlng) => {
  return latlng.lat <= (ALBERTA_CENTER.lat + ALBERTA_BOUNDS_PADDING_DEG) &&
         latlng.lat >= (ALBERTA_CENTER.lat - ALBERTA_BOUNDS_PADDING_DEG) &&
         latlng.lng <= (ALBERTA_CENTER.lng + ALBERTA_BOUNDS_PADDING_DEG) &&
         latlng.lng >= (ALBERTA_CENTER.lng - ALBERTA_BOUNDS_PADDING_DEG);
};

const placePredictionMarkers = (map, predictions) => {
  const regions = predictions?.regions ?? [];

  if (!regions.length) {
    updateStatus('Map ready. Add your datasets to highlight promising regions.');
    return;
  }

  const bounds = L.latLngBounds();

  regions.forEach((region) => {
    const confidence = Math.round(region.confidence * 100);
    const coordinates = L.latLng(region.coordinates);

    L.marker(coordinates)
      .addTo(map)
      .bindPopup(`
        <strong>${region.name}</strong><br />
        Estimated cloud seeding suitability: ${confidence}%
      `);

    bounds.extend(coordinates);
  });

  // Only fit bounds if every region lies within a loose bounding box around Alberta.
  const allWithinAlberta = regions.every(r => isWithinAlbertaBox(L.latLng(r.coordinates)));
  if (allWithinAlberta && regions.length) {
    map.fitBounds(bounds, { padding: [30, 30] });
  } else if (!allWithinAlberta) {
    // Keep original center, maybe zoom slightly if markers are far.
    console.log('Skipping fitBounds to keep Alberta focus. Some regions lie outside the Alberta focus area.');
  }
  updateStatus('Sample data displayed. Replace the API stub with your model outputs.');
};

const initialize = async () => {
  try {
    updateStatus('Loading map...');
    const map = initializeMap();

    updateStatus('Fetching sample predictions...');
    const predictions = await fetchPredictions();
    placePredictionMarkers(map, predictions);
  } catch (error) {
    console.error(error);
    updateStatus(error.message, true);
  }
};

initialize();
