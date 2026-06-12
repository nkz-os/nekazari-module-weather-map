export const METRICS = [
  { key: 'temperature_avg',     labelKey: 'metrics.temperature_avg',     unit: '°C' },
  { key: 'temperature_min',     labelKey: 'metrics.temperature_min',     unit: '°C' },
  { key: 'solar_radiation',     labelKey: 'metrics.solar_radiation',     unit: 'W/m²' },
  { key: 'eto',                 labelKey: 'metrics.eto',                 unit: 'mm/d' },
  { key: 'water_balance',       labelKey: 'metrics.water_balance',       unit: 'mm' },
  { key: 'frost_risk',          labelKey: 'metrics.frost_risk',          unit: '%' },
  { key: 'soil_moisture',       labelKey: 'metrics.soil_moisture',       unit: '% vol' },
] as const;

export type MetricKey = typeof METRICS[number]['key'];
