import * as L from '/vendor/leaflet/leaflet-src.esm.js';

const mapElement = document.getElementById('map');

const updateStatus = (message, isError = false) => {
  // Status updates removed - map now covers full screen
};

const fetchPredictions = async () => {
  const response = await fetch('/api/predictions');
  if (!response.ok) {
    throw new Error('Failed to load prediction data.');
  }

  return response.json();
};

const fetchCloudSeedingScores = async () => {
  const response = await fetch('/api/cloud-seeding');
  if (!response.ok) {
    throw new Error('Failed to load cloud seeding scores.');
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

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  // Store the setup function to be called after cloud seeding overlay is ready
  map._setupOverlay = () => {
    // Create black overlay covering everything except Alberta
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

        // Create black overlay that covers everything except Alberta
        if (albertaFeature) {
          // Create a world-covering polygon with Alberta as a hole
          const worldBounds = [
            [-90, -180], // Southwest corner of world
            [-90, 180],  // Southeast corner of world
            [90, 180],   // Northeast corner of world
            [90, -180],  // Northwest corner of world
            [-90, -180]  // Close the polygon
          ];

          // Extract Alberta coordinates (reverse to create hole)
          const albertaCoordinates = albertaFeature.geometry.coordinates;
          let albertaHoles = [];
          
          if (albertaFeature.geometry.type === 'Polygon') {
            // Single polygon - reverse the coordinates to create a hole
            albertaHoles = [albertaCoordinates[0].map(coord => [coord[1], coord[0]]).reverse()];
          } else if (albertaFeature.geometry.type === 'MultiPolygon') {
            // Multiple polygons - reverse each one to create holes
            albertaHoles = albertaCoordinates.map(polygon => 
              polygon[0].map(coord => [coord[1], coord[0]]).reverse()
            );
          }

          // Create polygon with holes (world polygon with Alberta cut out)
          const maskPolygon = L.polygon([worldBounds, ...albertaHoles], {
            color: 'transparent',
            weight: 0,
            fillColor: '#000000',
            fillOpacity: 1,
            interactive: false
          }).addTo(map);

          // Highlight Alberta outline on top of the mask
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
  };

  // Removed automatic user geolocation recentering per request.
  updateStatus('Map centered on Alberta, Canada.');

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

  // Markers removed - no location markers will be displayed
  // No bounds fitting needed since there are no markers
  updateStatus('Sample data displayed. Replace the API stub with your model outputs.');
};

const renderCloudSeedingOverlay = (map, payload) => {
  const gridPoints = payload?.data ?? [];
  if (!gridPoints.length) {
    console.warn('No cloud seeding data available for overlay.');
    return;
  }

  const step = payload?.meta?.step ?? 0.25;
  const halfStep = step / 2;
  const maxOpacity = 0.85;

  if (map.cloudSeedingLayer) {
    map.removeLayer(map.cloudSeedingLayer);
  }

  const layer = L.layerGroup();

  gridPoints.forEach(point => {
    const lat = Number(point?.latitude);
    const lon = Number(point?.longitude);
    const score = Number(point?.seeding_score);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return;
    }

    if (!Number.isFinite(score)) {
      return;
    }

    const clamped = Math.max(0, Math.min(1, score));

    const bounds = [
      [lat - halfStep, lon - halfStep],
      [lat + halfStep, lon + halfStep]
    ];

    const rect = L.rectangle(bounds, {
      stroke: false,
      fill: true,
      fillColor: '#2e7d32',
      fillOpacity: clamped * maxOpacity,
      interactive: false
    });

    layer.addLayer(rect);
  });

  if (layer.getLayers().length) {
    layer.addTo(map);
    map.cloudSeedingLayer = layer;
  }
};

const initialize = async () => {
  try {
    updateStatus('Loading map...');
    const map = initializeMap();

    updateStatus('Fetching sample predictions...');
    const predictions = await fetchPredictions();
    placePredictionMarkers(map, predictions);

    updateStatus('Applying cloud seeding overlay...');
    try {
      const cloudSeeding = await fetchCloudSeedingScores();
      renderCloudSeedingOverlay(map, cloudSeeding);
      // Add the black overlay after cloud seeding tiles so it appears on top
      map._setupOverlay();
      updateStatus('Sample data displayed with cloud seeding overlay.');
    } catch (overlayError) {
      console.error(overlayError);
      // Even if cloud seeding fails, we still want the black overlay
      map._setupOverlay();
      updateStatus('Sample data displayed. Cloud seeding overlay unavailable.');
    }
  } catch (error) {
    console.error(error);
    updateStatus(error.message, true);
  }
};

initialize();
