export const weatherMapSlots = {
  'map-layer': [{
    id: 'weather-map-raster-layer',
    moduleId: 'weather-map',
    component: 'WeatherRasterLayer',
    priority: 20,
  }],
  'layer-toggle': [{
    id: 'weather-map-toggle',
    moduleId: 'weather-map',
    component: 'WeatherLayerToggle',
    priority: 20,
  }],
  'context-panel': [{
    id: 'weather-map-control',
    moduleId: 'weather-map',
    component: 'WeatherLayerControl',
    priority: 20,
    showWhen: { entityType: ['AgriParcel'] },
  }],
  'entity-tree': [],
};
