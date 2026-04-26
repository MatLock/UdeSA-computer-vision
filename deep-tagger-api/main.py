import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.request.DeepTaggerRequest import DeepTaggerRequest
from model.response.DeepTaggerResponse import DeepTaggerResponse
from aux_functions.auxiliary import download_image
from machine_learning import k_means
from transformer import blip_transformer
from deep_learning import product_type_classifier
from deep_learning import multilabel_classifier
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
  img_array = download_image(image_url=request.image_url)
  product_type = product_type_classifier.predict(img_array=img_array)
  ml_tags = multilabel_classifier.predict(img_array=img_array,product_type=product_type)
  tags = {
    "color": "/".join(k_means.predict_k_colors(img_array=img_array)[0]),
    **ml_tags,
  }
  title = blip_transformer.predict_product_title(img_array=img_array)
  description = claude_client.generate_product_description(product_type=product_type,
                                                           title=title,
                                                           tags=tags)
  deep_tagger_response = DeepTaggerResponse(product_type=product_type,
                                          title=title,
                                          description=description,
                                          tags=tags)
  return deep_tagger_response

# test url https://i.postimg.cc/DwnSW-Dnh/test-shirt.avif
#https://www.militarykit.com/cdn/shop/files/mens-gildan-short-sleeve-heavy-cotton-tshirt-royal-blue.jpg?v=1766053652&width=1500
#https://http2.mlstatic.com/D_NQ_NP_2X_710446-MLA109522743040_042026-F.webp
#https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcSra_o0SB7t5Gb4GL5x-X7guFvalP2iVbJjPCHOs8623sGzIOP6MrQgs-A0T_go0cS2CCGNijpPC2dSsBI4PLCbYU8Cp_DVquVqbK7tcSOJ3AzBLoGzEKZv1Dw
#https://theblacktux.com/cdn/shop/files/black-patent-leather-shoes-204587.jpg?v=1741187566&width=1200
#https://i.ebayimg.com/images/g/RRAAAOSwl-hgXTJy/s-l1600.webp
#https://i.ebayimg.com/images/g/z2wAAeSwhhhp0BxR/s-l1600.webp
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8080)
