import React, { useCallback, useSyncExternalStore } from 'react';
import {
  getWeatherLayerState,
  setWeatherLayerState,
  subscribeWeatherLayer,
  type WeatherLayerState,
  type WeatherLayerStatus,
} from './weatherLayerStore';

export type { WeatherLayerStatus };

interface WeatherLayerControls extends WeatherLayerState {
  setMetric: (metric: string) => void;
  setDate: (date: string) => void;
  setVisible: (visible: boolean) => void;
  setOpacity: (opacity: number) => void;
  setStatus: (status: WeatherLayerStatus) => void;
}

export function useWeatherLayerContext(): WeatherLayerControls {
  const snap = useSyncExternalStore(
    subscribeWeatherLayer,
    getWeatherLayerState,
    getWeatherLayerState,
  );

  const setMetric = useCallback(
    (metric: string) => setWeatherLayerState({ metric, date: '', status: 'idle' }),
    [],
  );
  const setDate = useCallback((date: string) => setWeatherLayerState({ date }), []);
  const setVisible = useCallback((visible: boolean) => setWeatherLayerState({ visible }), []);
  const setOpacity = useCallback((opacity: number) => setWeatherLayerState({ opacity }), []);
  const setStatus = useCallback((status: WeatherLayerStatus) => setWeatherLayerState({ status }), []);

  return {
    ...snap,
    setMetric,
    setDate,
    setVisible,
    setOpacity,
    setStatus,
  };
}

/** Kept for module-kit withModuleProvider; state lives in weatherLayerStore. */
export function WeatherProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
