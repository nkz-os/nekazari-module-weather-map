import { defineModule } from '@nekazari/module-kit';
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
  slots: weatherMapSlots as never,
  data: {
    entities: [],
    timeseries: [],
  },
});
