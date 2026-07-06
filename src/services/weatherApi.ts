function apiBase(): string {
  return (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL
    || 'https://nkz.robotika.cloud';
}

export async function fetchLatestWeatherDate(metric: string): Promise<string | null> {
  const resp = await fetch(`${apiBase()}/api/weather-map/latest/${encodeURIComponent(metric)}`, {
    credentials: 'include',
    headers: { Accept: 'application/json' },
  });
  if (!resp.ok) return null;
  const data = (await resp.json()) as { date?: string };
  return data.date ?? null;
}
