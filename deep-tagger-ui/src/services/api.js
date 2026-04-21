const PREDICT_URL = 'http://localhost:8080/predict-image';

async function predictTags(imageUrl) {
  const response = await fetch(PREDICT_URL, {
    method: 'POST',
    headers: {
      'Accept': 'applicatiwon/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image_url: imageUrl }),
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  predictTags,
};
