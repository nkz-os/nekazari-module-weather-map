import React, { useEffect, useRef } from 'react';
import { useViewerOptional } from '@nekazari/sdk';
import { useWeatherLayerContext } from '../services/weatherLayerContext';
import { fetchLatestWeatherDate } from '../services/weatherApi';

const FRONTEND_HOST = 'https://nekazari.robotika.cloud';

const WeatherRasterLayer: React.FC = () => {
  const viewerCtx = useViewerOptional();
  const viewer = (viewerCtx as { cesiumViewer?: unknown })?.cesiumViewer;
  const tenantId =
    (viewerCtx as { tenantId?: string })?.tenantId ||
    (viewerCtx as { tenant?: string })?.tenant ||
    'default';

  const { metric, date, visible, opacity, setDate, setStatus } = useWeatherLayerContext();
  const layerRef = useRef<{ alpha?: number } | null>(null);
  const resolvedDateRef = useRef<string>('');

  useEffect(() => {
    if (!visible || date) return;
    let cancelled = false;
    setStatus('loading');
    fetchLatestWeatherDate(metric)
      .then((latest) => {
        if (cancelled) return;
        if (latest) {
          setDate(latest);
        } else {
          setStatus('empty');
        }
      })
      .catch(() => {
        if (!cancelled) setStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, [visible, date, metric, setDate, setStatus]);

  useEffect(() => {
    const CesiumLib = (window as { Cesium?: typeof import('cesium') }).Cesium;
    const v = viewer as {
      imageryLayers?: { remove: (l: unknown, destroy?: boolean) => void; addImageryProvider: (p: unknown) => { alpha?: number } };
      isDestroyed?: () => boolean;
    } | undefined;

    const removeLayer = () => {
      if (!layerRef.current || v?.isDestroyed?.()) return;
      try {
        v?.imageryLayers?.remove(layerRef.current, true);
      } catch {
        /* viewer torn down */
      }
      layerRef.current = null;
    };

    removeLayer();

    if (!viewer || !visible || !date || !CesiumLib) {
      if (!visible) setStatus('idle');
      return;
    }

    const pmtilesUrl = `${FRONTEND_HOST}/modules/weather-map/pmtiles/${tenantId}/${metric}/${date}.pmtiles`;
    let cancelled = false;
    setStatus('loading');
    resolvedDateRef.current = date;

    (async () => {
      try {
        const provider = await CesiumLib.createTileMapServiceImageryProvider({
          url: pmtilesUrl,
          minimumLevel: 10,
          maximumLevel: 15,
        });
        if (cancelled || v?.isDestroyed?.()) return;
        const layer = v?.imageryLayers?.addImageryProvider(provider);
        if (layer) {
          layer.alpha = opacity;
          layerRef.current = layer;
          setStatus('ready');
        }
      } catch (err) {
        console.error('[WeatherMap] PMTiles layer failed:', err);
        if (!cancelled) setStatus('empty');
      }
    })();

    return () => {
      cancelled = true;
      removeLayer();
    };
  }, [viewer, tenantId, metric, date, visible, opacity, setStatus]);

  useEffect(() => {
    if (layerRef.current) {
      layerRef.current.alpha = opacity;
    }
  }, [opacity]);

  return null;
};

export default WeatherRasterLayer;
