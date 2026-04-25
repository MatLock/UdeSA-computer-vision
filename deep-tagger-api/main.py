import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.request.DeepTaggerRequest import DeepTaggerRequest
from model.response.DeepTaggerResponse import DeepTaggerResponse
from aux_functions.aux import download_image
from machine_learning import k_means
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
    "color": "/".join(k_means.predict_k_colors(img_array)[0]),
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
# https://hips.hearstapps.com/vader-prod.s3.amazonaws.com/1736358500-baggy-jeans-compras-01-677eba422b2c3.jpg?crop=1xw:1xh;center,top&resize=980:*
# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGcQl0wyZ3PXFgTTYTfRHsK3YhvQ53i81ZQQ&s
#https://xcdn.next.co.uk/common/items/default/default/itemimages/3_4Ratio/product/lge/U88491s2.jpg?im=Resize,width=750
#https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcSra_o0SB7t5Gb4GL5x-X7guFvalP2iVbJjPCHOs8623sGzIOP6MrQgs-A0T_go0cS2CCGNijpPC2dSsBI4PLCbYU8Cp_DVquVqbK7tcSOJ3AzBLoGzEKZv1Dw
#https://theblacktux.com/cdn/shop/files/black-patent-leather-shoes-204587.jpg?v=1741187566&width=1200
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8080)
