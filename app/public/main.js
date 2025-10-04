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
    center: [39.8283, -98.5795], // Default center, will be updated if geolocation succeeds
    zoom: 4,
    zoomControl: false
  });

  L.control.zoom({ position: 'bottomright' }).addTo(map);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  // Try to get user's location
  if ('geolocation' in navigator) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        map.setView([latitude, longitude], 10);
        updateStatus('Map centered on your location.');
      },
      (error) => {
        console.warn('Geolocation error:', error.message);
        updateStatus('Using default map view.');
      }
    );
  }

  // Add click event listener to print coordinates and get cloud seeding likelihood
  map.on('click', async (event) => {
    const { lat, lng } = event.latlng;
    console.log(`Clicked location - Latitude: ${lat.toFixed(6)}, Longitude: ${lng.toFixed(6)}`);
    
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
  });

  return map;
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

  map.fitBounds(bounds, { padding: [30, 30] });
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
