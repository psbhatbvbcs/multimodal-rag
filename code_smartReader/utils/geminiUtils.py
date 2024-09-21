import google.generativeai as genai

from utils.tracker import store_object, fetch_object
import os

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure the Google Generative AI client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

"""
    Upload all images to the drive.
    - Used to send the image to the model after uploading in form of an image.
    - Ensures secure upload to model
"""


class GeminiUtils:
    def __init__(self):
        pass

    # Upload a file to Gemini. Necessary to pass to model
    def upload_to_gemini(self, path, mime_type="image/jpeg"):
        file = fetch_object(path)
        if file:
            return file
        else:
            file = genai.upload_file(path, mime_type=mime_type)
            store_object(file)
            return file

    # Upload all images in a directory to Gemini
    def upload_all_images(self, pdf_names):
        uploaded_images = []
        for pdf_name in pdf_names:
            pdf_base_name = os.path.splitext(os.path.basename(pdf_name))[0]
            input_dir = f"assets/Images/{pdf_base_name}"
            # List all JPG images in the directory
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(".jpg"):
                        image_path = os.path.join(root, file)
                        uploaded_file = self.upload_to_gemini(image_path)
                        uploaded_images.append(
                            (image_path, uploaded_file, pdf_name))

        return uploaded_images
