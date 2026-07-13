/**
 * Shared weather-map layer *configuration* across context-panel and map-layer
 * slots (each mounts in a separate React tree in the host viewer). Visibility,
 * opacity and load status now live in the host's unified LayerRegistry
 * (@nekazari/sdk useViewerLayer) — only the metric/date selection is module-local.
 */

export interface WeatherLayerState {
  metric: string;
  date: string;
}

let state: WeatherLayerState = {
  metric: 'temperature_avg',
  date: '',
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
