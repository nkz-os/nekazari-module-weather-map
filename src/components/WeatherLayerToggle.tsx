import React from 'react';
import { useTranslation } from '@nekazari/sdk';
import { SlotShellCompact } from '@nekazari/viewer-kit';
import { useWeatherLayerContext } from '../services/weatherLayerContext';
import { METRICS } from '../services/api';

const weatherAccent = { base: '#2563EB', soft: '#DBEAFE', strong: '#1E40AF' };

const WeatherLayerToggle: React.FC = () => {
  const { t } = useTranslation('weather-map');
  const {
    metric,
    setMetric,
    date,
    setDate,
    visible,
    setVisible,
    opacity,
    setOpacity,
    status,
  } = useWeatherLayerContext();

  const statusMsg =
    status === 'loading'
      ? t('layer.status.loading', 'Loading…')
      : status === 'empty'
        ? t('layer.status.empty', 'No raster data for this metric')
        : status === 'error'
          ? t('layer.status.error', 'Failed to load layer')
          : null;

  return (
    <SlotShellCompact moduleId="weather-map" accent={weatherAccent}>
      <div className="space-y-2 text-nkz-xs">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            role="switch"
            checked={visible}
            onChange={(e) => setVisible(e.target.checked)}
            className="cursor-pointer accent-nkz-accent-base"
          />
          <span className="font-medium">{t('layer.title', 'Weather')}</span>
        </label>

        {visible && (
          <div className="space-y-2 pl-1">
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              className="text-nkz-xs border border-nkz-border rounded-nkz-sm w-full bg-nkz-surface px-2 py-1"
            >
              {METRICS.map((m) => (
                <option key={m.key} value={m.key}>
                  {t(m.labelKey)} ({m.unit})
                </option>
              ))}
            </select>

            <div className="flex items-center gap-2">
              <span className="text-nkz-text-muted shrink-0">{t('controls.date', 'Date')}</span>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="text-nkz-xs border border-nkz-border rounded-nkz-sm flex-1 bg-nkz-surface px-2 py-1"
              />
            </div>

            <div className="flex items-center gap-2">
              <span className="text-nkz-text-muted shrink-0">{t('controls.opacity', 'Opacity')}</span>
              <input
                type="range"
                min={0.2}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="flex-1 accent-nkz-accent-base"
              />
              <span className="text-nkz-text-muted w-8 text-right">
                {Math.round(opacity * 100)}%
              </span>
            </div>

            {statusMsg && <p className="text-nkz-text-muted italic">{statusMsg}</p>}
          </div>
        )}
      </div>
    </SlotShellCompact>
  );
};

export default WeatherLayerToggle;
