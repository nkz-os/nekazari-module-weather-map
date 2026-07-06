import React from 'react';
import { useTranslation } from '@nekazari/sdk';
import { SlotShellCompact } from '@nekazari/viewer-kit';
import { useWeatherLayerContext } from '../services/weatherLayerContext';
import { METRICS } from '../services/api';

const weatherAccent = { base: '#2563EB', soft: '#DBEAFE', strong: '#1E40AF' };

const WeatherLayerControl: React.FC = () => {
  const { t } = useTranslation('weather-map');
  const { metric, setMetric, date, setDate, opacity, setOpacity, visible, setVisible } =
    useWeatherLayerContext();

  return (
    <SlotShellCompact moduleId="weather-map" accent={weatherAccent}>
      <div className="flex flex-col gap-nkz-tight text-nkz-sm">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={visible}
            onChange={(e) => setVisible(e.target.checked)}
            className="accent-nkz-accent-base"
          />
          <span>{t('controls.visible', 'Visible')}</span>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-nkz-xs text-nkz-text-muted">{t('controls.metric')}</span>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            className="text-nkz-sm border border-nkz-border rounded-nkz-sm bg-nkz-surface px-2 py-1"
          >
            {METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {t(m.labelKey)} ({m.unit})
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-nkz-xs text-nkz-text-muted">{t('controls.date')}</span>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="text-nkz-sm border border-nkz-border rounded-nkz-sm bg-nkz-surface px-2 py-1"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-nkz-xs text-nkz-text-muted">
            {t('controls.opacity')}: {Math.round(opacity * 100)}%
          </span>
          <input
            type="range"
            min={0.2}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="accent-nkz-accent-base"
          />
        </label>
      </div>
    </SlotShellCompact>
  );
};

export default WeatherLayerControl;
