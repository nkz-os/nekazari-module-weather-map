import React from 'react';
import { useTranslation } from '@nekazari/sdk';
import { SlotShellCompact } from '@nekazari/viewer-kit';
import { METRICS, type MetricKey } from '../services/api';

interface Props {
  metric: string;
  onMetricChange: (metric: string) => void;
  date?: string;
  onDateChange?: (date: string) => void;
  opacity: number;
  onOpacityChange: (opacity: number) => void;
}

const WeatherLayerControl: React.FC<Props> = ({
  metric,
  onMetricChange,
  date,
  onDateChange,
  opacity,
  onOpacityChange,
}) => {
  const { t } = useTranslation('weather-map');

  return (
    <SlotShellCompact moduleId="weather-map">
      <div className="weather-layer-control">
        {/* Metric selector */}
        <div className="weather-layer-control__field">
          <label className="weather-layer-control__label">
            {t('controls.metric')}
          </label>
          <select
            className="weather-layer-control__select"
            value={metric}
            onChange={(e) => onMetricChange(e.target.value)}
          >
            {METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {t(m.labelKey)} ({m.unit})
              </option>
            ))}
          </select>
        </div>

        {/* Date picker */}
        {onDateChange && (
          <div className="weather-layer-control__field">
            <label className="weather-layer-control__label">
              {t('controls.date')}
            </label>
            <input
              type="date"
              className="weather-layer-control__input"
              value={date ?? ''}
              onChange={(e) => onDateChange(e.target.value)}
            />
          </div>
        )}

        {/* Opacity slider */}
        <div className="weather-layer-control__field">
          <label className="weather-layer-control__label">
            {t('controls.opacity')}: {Math.round(opacity * 100)}%
          </label>
          <input
            type="range"
            className="weather-layer-control__slider"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => onOpacityChange(parseFloat(e.target.value))}
          />
        </div>
      </div>
    </SlotShellCompact>
  );
};

export default WeatherLayerControl;
