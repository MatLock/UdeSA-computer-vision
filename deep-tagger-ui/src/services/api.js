const PREDICT_URL = 'https://047282ea-cd44-496b-a00d-30af8e6cddcc.mock.pstmn.io/image';

async function predictTags(imageUrl) {
  const url = `${PREDICT_URL}?imageUrl=${encodeURIComponent(imageUrl)}`;
  const response = await fetch(url, { method: 'GET' });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  predictTags,
};
