from typing import Dict
from pydantic import BaseModel

class DeepTaggerResponse(BaseModel):
  product_type: str
  title: str
  description: str
  tags: Dict[str, str]