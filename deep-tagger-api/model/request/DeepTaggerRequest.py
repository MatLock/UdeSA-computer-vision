from pydantic import BaseModel


class DeepTaggerRequest(BaseModel):
  image_url: str