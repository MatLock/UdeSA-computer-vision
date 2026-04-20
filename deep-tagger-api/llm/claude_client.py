import os
import anthropic

_FALLBACK_DESCRIPTION = "Lorem ipsum dolor sit amet."

_api_key = os.environ.get("ANTHROPIC_API_KEY")
_client = anthropic.Anthropic(api_key=_api_key) if _api_key else None


def generate_product_description(product_type: str, title: str, tags: dict) -> str:
  if not _client:
    return _FALLBACK_DESCRIPTION

  tags_text = ", ".join(f"{k}: {v}" for k, v in tags.items())
  prompt = (
    f"Generate a short, compelling product description for an e-commerce listing.\n"
    f"Product type: {product_type}\n"
    f"Title: {title}\n"
    f"Tags: {tags_text}\n"
    f"Write only the description, 2-3 sentences max."
  )

  message = _client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt}],
  )
  return message.content[0].text