import uvicorn
from fastapi import FastAPI
from model.request.DeepTaggerRequest import DeepTaggerRequest
from model.response.DeepTaggerResponse import DeepTaggerResponse
from machine_learning import predict_k_colors

app = FastAPI()


@app.get("/")
async def root():
  return {"health": "UP"}


@app.post("/predict-image")
async def say_hello(request: DeepTaggerRequest) -> DeepTaggerResponse:
  tags = {
    "color": "Blue/Navy-Blue",
    "material": "Denim",
    "occasion": "Casual",
    "season": "Summer"
  }
  deep_tagger_response = DeepTaggerResponse(product_type='shirt',
                                          title='Blue Denim shirt',
                                          description='Comfortable and stylish blue denim shirt for everyday wear.',
                                          tags=tags)
  return deep_tagger_response


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8080)
