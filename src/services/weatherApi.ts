function apiBase(): string {
  return (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL
    || 'https://nkz.robotika.cloud';
}

function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    const keycloak = (window as { keycloak?: { token?: string } }).keycloak;
    if (keycloak?.token) return keycloak.token;
  } catch { /* keycloak not available */ }
  return null;
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function apiHeaders(): Record<string, string> {
  return { Accept: 'application/json', ...authHeaders() };
}

export async function fetchLatestWeatherDate(metric: string): Promise<string | null> {
  const resp = await fetch(`${apiBase()}/api/weather-map/latest/${encodeURIComponent(metric)}`, {
    credentials: 'include',
    headers: apiHeaders(),
  });
  if (!resp.ok) return null;
  const data = (await resp.json()) as { date?: string };
  return data.date ?? null;
}

export interface TileToken {
  tenant: string;
  token: string;
  expires: number;
}

export async function fetchTileToken(metric: string): Promise<TileToken | null> {
  const resp = await fetch(`${apiBase()}/api/weather-map/tiles-base/${encodeURIComponent(metric)}`, {
    credentials: 'include',
    headers: apiHeaders(),
  });
  if (!resp.ok) return null;
  return (await resp.json()) as TileToken;
}
