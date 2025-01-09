import base64
import io
import logging

from PIL import Image

# max size is 800x800 pixels
MAX_PIXELS = 800 * 800


def url_format_image(image_bytes: bytes) -> str:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/png;base64,{image_base64}"
    return image_data_url


def load_image(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return image_file.read()


def convert_image_type_to_png(image_path: str) -> bytes:
    with Image.open(image_path) as image:
        buffer = io.BytesIO()
        converted = image.convert("RGBA")
        converted.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()


def resize_image(image_bytes: bytes) -> bytes:
    output = image_bytes
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size

        total_pixels = width * height

        if total_pixels > MAX_PIXELS:
            scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            logging.info(f"Resizing image: original size: {width}x{height} new size: {new_width}x{new_height}")
            resized_image = image.resize((new_width, new_height))

            output_buffer = io.BytesIO()
            resized_image.save(output_buffer, format=image.format)
            output = output_buffer.getvalue()

    return output
