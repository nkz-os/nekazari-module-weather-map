import { defineModule, withModuleProvider } from '@nekazari/module-kit';
import { lazy } from 'react';
import './i18n';
import { weatherMapSlots } from './slots';
import pkg from '../package.json';

const WeatherMapMain = lazy(() => import('./WeatherMapApp'));

export default defineModule({
  id: 'weather-map',
  displayName: 'Weather Map',
  version: pkg.version,
  hostApiVersion: '^2.0.0',
  description: 'Weather-derived raster overlays — temperature, water balance, ET0, frost risk',
  accent: { base: '#2563EB', soft: '#DBEAFE', strong: '#1E40AF' },
  icon: 'cloud-sun',
  main: WeatherMapMain,
  slots: withModuleProvider(weatherMapSlots as never) as never,
  viewerLayers: [
    {
      id: 'weather-map-raster',
      titleKey: 'weather-map:layerTitle',
      supportsOpacity: true,
      defaultVisible: false,
    },
  ],
  data: {
    entities: ["AgriParcel", "AgriParcelRecord", "AgriSoil"],
    timeseries: ["AgriParcelRecord"],
  },
});
