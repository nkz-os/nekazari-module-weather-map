import React from 'react';
import { useTranslation } from '@nekazari/sdk';
import { Button } from '@nekazari/ui-kit';

interface Props {
  visible: boolean;
  onToggle: () => void;
}

const WeatherLayerToggle: React.FC<Props> = ({ visible, onToggle }) => {
  const { t } = useTranslation('weather-map');

  return (
    <Button
      variant={visible ? 'primary' : 'secondary'}
      onClick={onToggle}
      size="small"
    >
      {t('layerToggle')}
    </Button>
  );
};

export default WeatherLayerToggle;
