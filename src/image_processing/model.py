import json
import logging
import os

import pydantic
import requests
from dotenv import load_dotenv

from src.image_processing import image_conversion

load_dotenv()
TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
AI_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Request parameters
MAX_RETURNED_TOKENS = 1000
PROMPT = "Caption the image, include any known people and text present"


class Image(pydantic.BaseModel):
    url: str
    name: str
    path: str


images = [
    Image(
        url="https://images.unsplash.com/photo-1513639776629-7b61b0ac49cb?q=80&w=2067&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        name="KFC",
        path="../../images/brand.jpg",
    ),
    Image(
        url="https://media.licdn.com/dms/image/v2/D4E12AQFVpgPdr24kKg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1715607879180?e=2147483647&v=beta&t=DEI1-Xnbk5YESFqQ5Qr5o3-90_zfjz0eCB908KEqNq4",
        name="Famous alien",
        path="../../images/famous_alien.png",
    ),
    Image(
        url="https://static.wikia.nocookie.net/704fbcc8-a16b-4005-aa8b-d33e1a0fd8c2/scale-to-width/493",
        name="Overly attached girlfriend meme",
        path="../../images/girlfriend_meme.png",
    ),
    Image(
        url="https://images.unsplash.com/photo-1580130857334-2f9b6d01d99d?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        name="Uncle Sam",
        path="../../images/uncle_sam.png",
    )
]


def create_image_payload(image_path: str) -> str:
    if not image_path.endswith(".png"):
        image_bytes = image_conversion.convert_image_type_to_png(image_path)
    else:
        image_bytes = image_conversion.load_image(image_path)

    resized_image_bytes = image_conversion.resize_image(image_bytes)

    image_data_url = image_conversion.url_format_image(resized_image_bytes)
    return image_data_url


def evaluate_image(image_data_url: str, prompt: str) -> str:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        "max_tokens": MAX_RETURNED_TOKENS,
    }
    response = make_request(payload)

    return response["choices"][0]["message"]["content"]


def make_request(payload: dict) -> dict:
    url = f"https://api-inference.huggingface.co/models/{AI_MODEL_NAME}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        logging.error(f"Failed to make request: {response}")
    return json.loads(response.content.decode("utf-8"))


if __name__ == "__main__":
    image_url = create_image_payload("../../images/girlfriend_meme2.png")
    inferred_context = evaluate_image(image_url, PROMPT)
    print(inferred_context)
