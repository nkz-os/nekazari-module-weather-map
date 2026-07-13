import React, { useCallback, useSyncExternalStore } from 'react';
import {
  getWeatherLayerState,
  setWeatherLayerState,
  subscribeWeatherLayer,
  type WeatherLayerState,
} from './weatherLayerStore';

interface WeatherLayerControls extends WeatherLayerState {
  setMetric: (metric: string) => void;
  setDate: (date: string) => void;
}

export function useWeatherLayerContext(): WeatherLayerControls {
  const snap = useSyncExternalStore(
    subscribeWeatherLayer,
    getWeatherLayerState,
    getWeatherLayerState,
  );

  const setMetric = useCallback(
    (metric: string) => setWeatherLayerState({ metric, date: '' }),
    [],
  );
  const setDate = useCallback((date: string) => setWeatherLayerState({ date }), []);

  return {
    ...snap,
    setMetric,
    setDate,
  };
}

/** Kept for module-kit withModuleProvider; state lives in weatherLayerStore. */
export function WeatherProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
