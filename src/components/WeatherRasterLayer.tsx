import React, { useEffect, useMemo } from 'react';
import { useViewer } from '@nekazari/sdk';

interface Props {
  metric: string;
  date?: string;
  opacity?: number;
}

const API_BASE = (import.meta as any).env?.VITE_API_URL || 'https://nkz.robotika.cloud';

const WeatherRasterLayer: React.FC<Props> = ({ metric, date, opacity = 0.7 }) => {
  const { cesiumViewer: viewer } = useViewer();

  const url = useMemo(() => {
    const params = new URLSearchParams();
    if (date) params.set('date', date);
    const qs = params.toString();
    return `${API_BASE}/api/weather-map/tiles/${metric}/{z}/{x}/{y}.png${qs ? '?' + qs : ''}`;
  }, [metric, date]);

  useEffect(() => {
    if (!viewer) return;

    const provider = new (Cesium as any).UrlTemplateImageryProvider({ url });
    const layer = viewer.imageryLayers.addImageryProvider(provider);
    layer.alpha = opacity;

    return () => {
      viewer.imageryLayers.remove(layer);
    };
  }, [viewer, url, opacity]);

  return null;
};

export default WeatherRasterLayer;
