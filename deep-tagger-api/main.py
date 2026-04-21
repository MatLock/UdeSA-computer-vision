import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.request.DeepTaggerRequest import DeepTaggerRequest
from model.response.DeepTaggerResponse import DeepTaggerResponse
from aux_functions.aux import download_image
from machine_learning import knn_model
from transformer import blip_transformer
from deep_learning import product_type_classifier
from llm import claude_client

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000"],
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.get("/")
async def root():
  return {"health": "UP"}


@app.post("/predict-image")
def predict_from_image(request: DeepTaggerRequest) -> DeepTaggerResponse:
  img_array = download_image(request.image_url)
  tags = {
    "color": "/".join(knn_model.predict_k_colors(img_array)[0]),
    "material": "Denim",
    "occasion": "Casual",
    "season": "Summer"
  }
  product_type = product_type_classifier.predict(img_array)
  title = blip_transformer.predict_product_title(img_array)
  description = claude_client.generate_product_description(product_type, title, tags)
  deep_tagger_response = DeepTaggerResponse(product_type=product_type,
                                          title=title,
                                          description=description,
                                          tags=tags)
  return deep_tagger_response

# test url https://i.postimg.cc/DwnSW-Dnh/test-shirt.avif

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8080)
