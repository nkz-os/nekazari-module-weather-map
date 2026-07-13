import React, { useEffect, useRef } from 'react';
import { useViewerOptional, useViewerLayer } from '@nekazari/sdk';
import { useWeatherLayerContext } from '../services/weatherLayerContext';
import { fetchLatestWeatherDate } from '../services/weatherApi';

const API_BASE =
  (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL
  || 'https://nkz.robotika.cloud';
// Backend COGs are generated at a single zoom level (14); Cesium upsamples
// beyond it and misses (transparent 404s) below it.
const TILE_LEVEL = 14;

const WeatherRasterLayer: React.FC = () => {
  const viewerCtx = useViewerOptional();
  const viewer = (viewerCtx as { cesiumViewer?: unknown })?.cesiumViewer;

  const { metric, date, setDate } = useWeatherLayerContext();
  // Visibility, opacity (0–100) and load status are driven by the host's
  // unified Layers menu via the shared LayerRegistry.
  const { visible, opacity, setStatus } = useViewerLayer('weather-map-raster');
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

    let cancelled = false;
    setStatus('loading');
    resolvedDateRef.current = date;

    try {
      // Tile requests are plain XHR from Cesium: mark the API host as trusted
      // so the nkz_token cookie travels with them (require_tenant on backend).
      const apiHost = new URL(API_BASE).hostname;
      CesiumLib.TrustedServers.add(apiHost, 443);

      const provider = new CesiumLib.UrlTemplateImageryProvider({
        url: `${API_BASE}/api/weather-map/tiles/${encodeURIComponent(metric)}/{z}/{x}/{y}.png?date=${encodeURIComponent(date)}`,
        tilingScheme: new CesiumLib.WebMercatorTilingScheme(),
        maximumLevel: TILE_LEVEL,
        // Cut global 404 spam: weather COGs only cover EU tenants.
        rectangle: CesiumLib.Rectangle.fromDegrees(-11.0, 34.0, 32.0, 62.0),
        enablePickFeatures: false,
      });
      if (cancelled || v?.isDestroyed?.()) return;
      const layer = v?.imageryLayers?.addImageryProvider(provider);
      if (layer) {
        layer.alpha = opacity / 100;
        layerRef.current = layer;
        setStatus('ready');
      }
    } catch (err) {
      console.error('[WeatherMap] tile layer failed:', err);
      if (!cancelled) setStatus('empty');
    }

    return () => {
      cancelled = true;
      removeLayer();
    };
  }, [viewer, metric, date, visible, opacity, setStatus]);

  useEffect(() => {
    if (layerRef.current) {
      layerRef.current.alpha = opacity / 100;
    }
  }, [opacity]);

  return null;
};

export default WeatherRasterLayer;
