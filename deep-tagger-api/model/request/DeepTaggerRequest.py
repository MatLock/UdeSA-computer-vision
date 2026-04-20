from pydantic import BaseModel, Field


class DeepTaggerRequest(BaseModel):
  image_url: str = Field(examples=["https://www.militarykit.com/cdn/shop/files/mens-gildan-short-sleeve-heavy-cotton-tshirt-royal-blue.jpg?v=1766053652&width=1500"])