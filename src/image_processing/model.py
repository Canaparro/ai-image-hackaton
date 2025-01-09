import base64
import json
import logging
import os

import requests
from dotenv import load_dotenv
from requests import Response

load_dotenv()
TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
AI_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def create_image_payload(image_path: str) -> str:
    image_bytes = load_image(image_path)
    image_data_url = format_image(image_bytes)
    return image_data_url


def format_image(image_bytes: bytes) -> str:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/png;base64,{image_base64}"
    return image_data_url


def load_image(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return image_file.read()


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
    image = create_image_payload("../../images/famous_alien.png")
    result = evaluate_image(image, "describe what you see in the image, including known people")
    print(result)
