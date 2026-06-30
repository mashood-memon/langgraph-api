import voyageai
import os
from PIL import Image


from app.config import get_settings

_s = get_settings()

vo = voyageai.Client(api_key=_s.voyage_api_key)

MODEL = "voyage-multimodal-3.5"


def embed_text(texts: list[str], input_type: str = "document", batch_size: int = 100) -> list[list[float]]:
    """Embed text chunks in batches to avoid Voyage API limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = vo.multimodal_embed(
            inputs=[[t] for t in batch],
            model=MODEL,
            input_type=input_type,
        )
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def embed_images(images: list[Image.Image], input_type: str = "document", batch_size: int = 10) -> list[list[float]]:
    """Embed PIL images in batches to avoid Voyage payload size limits."""
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        result = vo.multimodal_embed(
            inputs=[[img] for img in batch],
            model=MODEL,
            input_type=input_type,
        )
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Single query embedding — searches both text and image content in the same collection."""
    result = vo.multimodal_embed(
        inputs=[[query]],
        model=MODEL,
        input_type="query",
    )
    return result.embeddings[0]


def render_pdf_pages(pdf_path: str) -> list[Image.Image]:
    """Render each PDF page as a PIL image, for the image embedding path."""
    from pdf2image import convert_from_path
    return convert_from_path(pdf_path, dpi=150)