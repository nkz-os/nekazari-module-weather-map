import React from 'react';
import WeatherRasterLayer from './components/WeatherRasterLayer';
import WeatherLayerControl from './components/WeatherLayerControl';

const WeatherMapApp: React.FC = () => {
  return (
    <>
      <WeatherLayerControl />
      <WeatherRasterLayer />
    </>
  );
};

export default WeatherMapApp;
