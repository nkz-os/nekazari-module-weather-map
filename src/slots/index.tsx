import React from 'react';
import WeatherRasterLayer from '../components/WeatherRasterLayer';
import WeatherLayerControl from '../components/WeatherLayerControl';
import { WeatherProvider } from '../services/weatherLayerContext';

const MODULE_ID = 'weather-map';

export const weatherMapSlots = {
  'map-layer': [
    {
      id: 'weather-map-raster-layer',
      moduleId: MODULE_ID,
      component: 'WeatherRasterLayer',
      priority: 20,
      localComponent: WeatherRasterLayer,
    },
  ],
  'context-panel': [
    {
      id: 'weather-map-control',
      moduleId: MODULE_ID,
      component: 'WeatherLayerControl',
      priority: 20,
      localComponent: WeatherLayerControl,
      showWhen: { entityType: ['AgriParcel'] },
    },
  ],
  'entity-tree': [],
  moduleProvider: WeatherProvider,
};
