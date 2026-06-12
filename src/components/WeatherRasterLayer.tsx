import React, { useEffect, useMemo } from 'react';
import { useConfig } from '@nekazari/sdk';
import { useViewer } from '@nekazari/viewer-kit';

interface Props {
  metric: string;
  date?: string;
  opacity?: number;
}

const WeatherRasterLayer: React.FC<Props> = ({ metric, date, opacity = 0.7 }) => {
  const { baseUrl } = useConfig();
  const viewer = useViewer();

  const url = useMemo(() => {
    const params = new URLSearchParams();
    if (date) params.set('date', date);
    const qs = params.toString();
    return `${baseUrl}/api/weather-map/tiles/${metric}/{z}/{x}/{y}.png${qs ? '?' + qs : ''}`;
  }, [baseUrl, metric, date]);

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
