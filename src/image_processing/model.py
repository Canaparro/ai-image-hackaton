import json
import logging
import os

import requests
from dotenv import load_dotenv
from requests import Response

from src.image_processing import image_conversion

load_dotenv()
TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
AI_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def create_image_payload(image_path: str) -> str:
    if not image_path.endswith(".png"):
        image_bytes = image_conversion.convert_image_type_to_png(image_path)
    else:
        image_bytes = image_conversion.load_image(image_path)

    resized_image_bytes = image_conversion.resize_image(image_bytes)

    image_data_url = image_conversion.url_format_image(resized_image_bytes)
    return image_data_url


def evaluate_image(image_data_url: str, prompt: str) -> str:
    # TODO: model is returning only 148 tokens, so it is cutting off some responses
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ]
    }
    response = make_request(payload)

    result = json.loads(response.content)
    return result["choices"][0]["message"]["content"]


def make_request(payload: dict) -> Response:
    url = f"https://api-inference.huggingface.co/models/{AI_MODEL_NAME}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        logging.error(f"Failed to make request: {response}")
    return json.loads(response.content.decode("utf-8"))


if __name__ == "__main__":
    image = create_image_payload("../../images/brand.jpg")
    inferred_context = evaluate_image(image, "describe what you see in the image, including known people")
    print(inferred_context)
