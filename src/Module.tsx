import React from 'react';
import { useTranslation } from '@nekazari/sdk';

const WeatherMapModule: React.FC = () => {
  const { t } = useTranslation('weather-map');
  return (
    <div>
      <h2>{t('title')}</h2>
      <p>{t('description')}</p>
    </div>
  );
};

export default WeatherMapModule;
