import React, { useState } from 'react';
import { useTranslation } from '@nekazari/sdk';
import WeatherRasterLayer from './components/WeatherRasterLayer';
import WeatherLayerControl from './components/WeatherLayerControl';
import WeatherLayerToggle from './components/WeatherLayerToggle';

export const mapLayer = WeatherRasterLayer;
export const contextPanel = WeatherLayerControl;
export const layerToggle = WeatherLayerToggle;

const Module: React.FC = () => {
  const [metric, setMetric] = useState('temperature_avg');
  const [visible, setVisible] = useState(true);
  const [opacity, setOpacity] = useState(0.7);
  const { t } = useTranslation('weather-map');

  return (
    <>
      <WeatherLayerControl
        metric={metric}
        onMetricChange={setMetric}
        opacity={opacity}
        onOpacityChange={setOpacity}
      />
      <WeatherLayerToggle visible={visible} onToggle={() => setVisible(!visible)} />
      {visible && (
        <WeatherRasterLayer metric={metric} opacity={opacity} />
      )}
    </>
  );
};

export default Module;
