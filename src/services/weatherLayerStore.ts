/**
 * Shared weather-map layer state across layer-toggle, map-layer and module page
 * (each slot mounts in a separate React tree in the host viewer).
 */

export type WeatherLayerStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'error';

export interface WeatherLayerState {
  metric: string;
  date: string;
  visible: boolean;
  opacity: number;
  status: WeatherLayerStatus;
}

let state: WeatherLayerState = {
  metric: 'temperature_avg',
  date: '',
  visible: false,
  opacity: 0.7,
  status: 'idle',
};

const listeners = new Set<() => void>();

function emit(): void {
  listeners.forEach((l) => l());
}

export function getWeatherLayerState(): WeatherLayerState {
  return state;
}

export function setWeatherLayerState(patch: Partial<WeatherLayerState>): void {
  state = { ...state, ...patch };
  emit();
}

export function subscribeWeatherLayer(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
