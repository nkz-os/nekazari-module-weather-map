import React, { useEffect, useMemo } from 'react';
import { useViewer } from '@nekazari/sdk';

interface Props {
  metric: string;
  date?: string;
  opacity?: number;
  tenantId?: string;
}

const FRONTEND_HOST = 'https://nekazari.robotika.cloud';

const WeatherRasterLayer: React.FC<Props> = ({ metric, date, opacity = 0.7, tenantId }) => {
  const viewerCtx = useViewer();
  const viewer = (viewerCtx as any).cesiumViewer;
  const CesiumLib = (window as any).Cesium;

  // Derive tenantId from viewer context if not explicitly provided
  const resolvedTenantId = tenantId || (viewerCtx as any).tenantId || 'default';

  const pmtilesUrl = useMemo(() => {
    if (!date) return null; // Need a date to resolve the PMTiles
    return `${FRONTEND_HOST}/modules/weather-map/pmtiles/${resolvedTenantId}/${metric}/${date}.pmtiles`;
  }, [metric, date, resolvedTenantId]);

  useEffect(() => {
    if (!viewer || !pmtilesUrl || !CesiumLib) return;

    let layer: any = null;

    (async () => {
      try {
        // Use Cesium's PMTiles support via createTileMapServiceImageryProvider
        // PMTiles implements the Tile Map Service specification
        const provider = await CesiumLib.createTileMapServiceImageryProvider({
          url: pmtilesUrl,
          minimumLevel: 10,
          maximumLevel: 15,
        });

        layer = viewer.imageryLayers.addImageryProvider(provider);
        layer.alpha = opacity;
      } catch (err) {
        console.error('[WeatherMap] PMTiles layer failed:', err);
      }
    })();

    return () => {
      if (layer) {
        viewer.imageryLayers.remove(layer);
      }
    };
  }, [viewer, pmtilesUrl, opacity, CesiumLib]);

  // Show nothing if no date is selected
  if (!date || !pmtilesUrl) return null;

  return null;
};

export default WeatherRasterLayer;
