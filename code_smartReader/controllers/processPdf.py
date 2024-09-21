import os
import json
import io
import time
import tempfile
import numpy as np
import csv

import tabula
from PIL import Image
from PyPDF2 import PdfReader
from pypdf import PdfReader as ImageReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import google.generativeai as genai

from utils.modelSettings import generation_config, safety_settings
from utils.modelInstructions import image_retriever_instruction
from utils.geminiUtils import GeminiUtils

# Configure the API key for Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class PDFprocessor:
    """
    A class for processing PDF documents, extracting text, tables, and images, and generating summaries.
    """

    def __init__(self, pdf_docs, pdf_names):
        """
        Initialize the PDF processor with provided PDF documents and names.

        Args:
            pdf_docs (list): List of PDF documents.
            pdf_names (list): List of PDF document names.
        """
        self.pdf_docs = pdf_docs
        self.pdf_names = pdf_names

    """
        MAIN FUNCTION 1 : parse_pdf()
        - Parses PDF for text, converts texts to chunks
        - Uses functions listed below:
            - get_pdf_text()
            - get_text_chunks()
    """

    def parse_pdf(self):
        """
        Parses the PDF to get text and then splits it into chunks.

        Returns:
            list: List of text chunks.
        """
        text_with_metadata = self.get_pdf_text()
        chunks = self.get_text_chunks(
            text_with_metadata, type="text", csv_name="")
        return chunks

    """
        MAIN FUNCTION 2 : get_tables()
        - Parses PDF for tables, converts all tables to csv and then to chunks
        - Uses functions listed below:
            - get_csv_string()
            - filter_csv()
            - get_text_chunks()
    """

    def get_tables(self):
        """
        Extracts tables from the PDFs and splits them into chunks.

        Returns:
            list: List of text chunks from tables.
        """
        csv_chunks = []
        for pdf, pdf_name in zip(self.pdf_docs, self.pdf_names):
            tabula.convert_into(pdf, "context_docs/output.csv",
                                output_format="csv", pages="all", lattice=True)

            self.filter_csv("context_docs/output.csv",
                            "context_docs/output_filtered.csv")

            csv_string = self.get_csv_string(
                "context_docs/output_filtered.csv")
            csv_chunks.extend(self.get_text_chunks(
                text_with_metadata=csv_string, type="csv", csv_name=pdf_name))
        return csv_chunks

    """
        MAIN FUNCTION 3 : extract_images()
        - Parses PDF for all images, stores them in proper folders
    """

    def extract_images(self):
        """Extracts images from the PDFs and returns them as a list of PIL images."""
        try:
            min_width = 50
            min_height = 50

            for pdf, pdf_name in zip(self.pdf_docs, self.pdf_names):
                reader = ImageReader(pdf)
                pdf_base_name = os.path.splitext(os.path.basename(pdf_name))[0]

                output_dir = f"assets/Images/{pdf_base_name}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                image_index = 0

                # Iterate over all pages and extract images
                for index, page in enumerate(reader.pages):
                    try:
                        images = page.images
                        print(f"Page {index}: Found {len(images)} images")
                    except Exception as e:
                        print(
                            f"Error extracting images from page {index}: {e}")
                        continue

                    try:
                        for image in images:
                            try:
                                # Log the image extraction attempt
                                print(f"Processing image on page {index}")
                                image_data = io.BytesIO(image.data)
                                try:
                                    data = Image.open(image_data)
                                except all:
                                    print(
                                        f"Skipping unsupported image on page {index}")
                                    continue
                                with Image.open(image_data) as img:
                                    if img.width < min_width or img.height < min_height:
                                        print(
                                            f"Skipping small image on page {index}: {img.width}x{img.height}")
                                        continue
                                    if self.is_blank_image(img):
                                        print(
                                            f"Skipping blank image on page {index}")
                                        continue
                                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                                    if img.mode in ("RGBA", "LA", "P"):
                                        img = img.convert("RGB")
                                    image_name = f"{pdf_base_name}_image_{index}_{image_index}.jpg"
                                    output_path = os.path.join(
                                        output_dir, image_name)
                                    img.save(output_path, "JPEG")
                                    image_index += 1
                            except ValueError as ve:
                                print(
                                    f"ValueError processing image on page {index}: {ve}")
                                continue
                            except Exception as e:
                                print(
                                    f"Error processing image on page {index}: {e}")
                    except Exception as e:
                        print(
                            f"Error accessing image data on page {index}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting images: {e}")

    """
        MAIN FUNCTION 4 : get_image_chunks()
        - Used in APPROACH 2
        - Gets all descriptions of images, converts them to chunks
        - Uses functions listed below:
            - upload_all_images()
    """

    def get_image_chunks(self):
        """
        Uploads images to Gemini and processes them to generate summaries.

        Returns:
            list: List of Document objects containing image summaries.
        """
        try:
            geminiUtils = GeminiUtils()
            uploaded_images = geminiUtils.upload_all_images(
                pdf_names=self.pdf_names)

            def process_batch(batch):
                history_parts = []
                for image_path, file, pdf_name in batch:
                    history_parts.extend([f"{image_path}: ", file])

                chat_history = [{"role": "user", "parts": history_parts}]
                chat_session = self.initialize_image_ai(history=chat_history)
                if chat_session:
                    image_response = chat_session.send_message("Analyse")

                if image_response:
                    response_texts = []
                    for response in image_response:
                        if hasattr(response, 'parts'):
                            for part in response.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_texts.append(part.text)

                if response_texts:
                    json_response_text = response_texts[0]
                    json_response_text = json_response_text.strip().strip('```json').strip('```')
                    image_summaries = json.loads(json_response_text)
                    docs = []
                    for image_path_with_colon, summary in image_summaries.items():
                        image_path = image_path_with_colon.strip(': ')
                        pdf_name = next(pdf_name for path, _,
                                        pdf_name in batch if path == image_path)
                        doc = Document(page_content=f'{summary}', metadata={
                            "path": image_path, "doc_name": pdf_name})
                        docs.append(doc)
                    return docs

            all_docs = []
            batch_size = 5
            for i in range(0, len(uploaded_images), batch_size):
                batch = uploaded_images[i:i + batch_size]
                docs = process_batch(batch)
                if docs:
                    all_docs.extend(docs)

                if (i // batch_size + 1) % 6 == 0:
                    time.sleep(60)

            return all_docs
        except Exception as e:
            print(e)
            return None

    """
        MAIN FUNCTION 5 : embed_images()
        - Used in both approaches
        - Converts all images to embeddings with MediaPipe library. Saves the embeddings in CSV. 
        - Embeddings are features in an image
    """

    def embed_images(self, model_path='./assets/models/embedder.tflite', output_csv_path='context_docs/img_embeddings.csv'):
        """
        Embeds images using a pre-trained model and saves embeddings to a CSV file.

        Args:
            model_path (str): Path to the Image Embedder model file.
            output_csv_path (str): Path to save the embeddings as a CSV file.
        """
        # Create options for Image Embedder
        base_options = python.BaseOptions(model_asset_path=model_path)
        l2_normalize = True
        quantize = False
        options = vision.ImageEmbedderOptions(
            base_options=base_options, l2_normalize=l2_normalize, quantize=quantize
        )
        embeddings = []
        metadata = {}
        image_filenames = []

        for pdf_name in self.pdf_names:
            pdf_base_name = os.path.splitext(os.path.basename(pdf_name))[0]
            image_directory = f"assets/Images/{pdf_base_name}"
            for f in os.listdir(image_directory):
                if os.path.isfile(os.path.join(image_directory, f)):
                    image_filenames.append(os.path.join(image_directory, f))

        # Create Image Embedder
        with vision.ImageEmbedder.create_from_options(options) as embedder:
            for i, filename in enumerate(image_filenames):
                try:
                    image = mp.Image.create_from_file(filename)
                    embedding_result = embedder.embed(image)
                    embedding_vector = np.array(
                        embedding_result.embeddings[0].embedding).astype('float32')
                    embeddings.append((filename, embedding_vector.tolist()))
                    metadata[i] = {'filename': filename}
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        print(f"Number of embeddings: {len(embeddings)}")

        # Save embeddings to CSV file
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for filename, embedding_vector in embeddings:
                csvwriter.writerow([filename] + embedding_vector)

    """
        MAIN FUNCTION 6 : calculate_similarity()
        - Used in both approaches
        - Calculates similarity between the image a user uploads and the extracted images from the PDF.
        - Obtaining the image with the max similarity score. Uses cosine_similarity
    """

    def calculate_similarity(self, uploaded_image, model_path='./assets/models/embedder.tflite',
                             embeddings_csv_path='context_docs/img_embeddings.csv'):
        """
        Calculates the similarity between an uploaded image and the embedded images in a CSV file.

        Args:
            uploaded_image: The image to compare.
            model_path (str): Path to the Image Embedder model file.
            embeddings_csv_path (str): Path to the CSV file containing image embeddings.

        Returns:
            str: Path to the image with the highest similarity to the uploaded image.
        """
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.getbuffer())
            temp_file_path = temp_file.name

        # Create options for Image Embedder
        base_options = python.BaseOptions(model_asset_path=model_path)
        l2_normalize = True
        quantize = False
        options = vision.ImageEmbedderOptions(
            base_options=base_options, l2_normalize=l2_normalize, quantize=quantize
        )

        # Embed the uploaded image
        with vision.ImageEmbedder.create_from_options(options) as embedder:
            uploaded_image_mp = mp.Image.create_from_file(temp_file_path)
            uploaded_image_embedding = embedder.embed(
                uploaded_image_mp).embeddings[0].embedding

            # Initialize variables to track highest similarity and corresponding image path
            highest_similarity = -1
            most_similar_image_path = None

            # Load embeddings from CSV and compare
            with open(embeddings_csv_path, newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    image_path = row[0]
                    embedding_vector = np.array(
                        list(map(float, row[1:]))).astype('float32')

                    similarity = np.dot(uploaded_image_embedding, embedding_vector) / (
                        np.linalg.norm(uploaded_image_embedding) * np.linalg.norm(embedding_vector))
                    print(f"Similarity with {image_path}: {similarity}")

                    # Update highest similarity and most similar image path if needed
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_image_path = image_path
                return most_similar_image_path
        return None

    def get_pdf_text(self):
        """
        Extracts text from each page of the PDF with page numbers.

        Returns:
            list: List of dictionaries containing document name, page number, and extracted text.
        """
        try:
            text_with_metadata = []
            for pdf, pdf_name in zip(self.pdf_docs, self.pdf_names):
                pdf_reader = PdfReader(pdf)
                for page_num, page in enumerate(pdf_reader.pages, start=0):
                    text = page.extract_text()
                    if text:
                        text_with_metadata.append(
                            {"doc_name": pdf_name, "page_num": page_num, "text": text})
            return text_with_metadata
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {e}")

    def get_text_chunks(self, text_with_metadata, type, csv_name):
        """
        Splits the raw text into chunks, preserving page metadata.

        Args:
            text_with_metadata (list): List of text with metadata.
            type (str): Type of document (text or csv).
            csv_name (str): Name of the CSV file.

        Returns:
            list: List of text chunks.
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            if type == "csv":
                chunks = text_splitter.create_documents([text_with_metadata])
                for doc in chunks:
                    doc.metadata["doc_name"] = csv_name
                return chunks
            else:
                chunks = []
                for item in text_with_metadata:
                    pdf_name = item["doc_name"]
                    page_num = item["page_num"]
                    raw_text = item["text"]
                    docs = text_splitter.create_documents([raw_text])
                    for doc in docs:
                        doc.metadata["doc_name"] = pdf_name
                        doc.metadata["page_num"] = page_num
                        chunks.append(doc)
                return chunks
        except Exception as e:
            raise RuntimeError(f"Error splitting text: {e}")

    def filter_csv(self, input_csv_path, output_csv_path):
        """
        Filters CSV rows to remove those with insufficient data.

        Args:
            input_csv_path (str): Path to the input CSV file.
            output_csv_path (str): Path to the output filtered CSV file.
        """
        try:
            with open(input_csv_path, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)

            with open(output_csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for row in rows:
                    if sum(1 for column in row if column.strip()) <= 2:
                        continue
                    if any(column.strip() for column in row[1:]):
                        writer.writerow(row)
        except Exception as e:
            raise RuntimeError(f"Error filtering CSV: {e}")

    def get_csv_string(self, input_csv_path):
        """
        Converts a CSV file to a string representation suitable for further processing.

        Args:
            input_csv_path (str): Path to the input CSV file.

        Returns:
            str: String representation of the CSV file.
        """
        try:
            bigstring = []
            with open(input_csv_path, 'r') as f:
                lines = f.readlines()
                smallstring = ""
                for line in lines:
                    new = line.strip()
                    new = new.replace('"', '')
                    smallstring += new
                    if new.find(",,,,,,,,,,,") != -1:
                        bigstring.append(smallstring)
                        smallstring = ""
                bigstring.append(smallstring)

            bigstring_str = repr(bigstring)
            return bigstring_str
        except Exception as e:
            raise RuntimeError(f"Error getting CSV string: {e}")

    def is_blank_image(self, image):
        """
        Checks if an image is blank.

        Args:
            image (PIL.Image.Image): The image to check.

        Returns:
            bool: True if the image is blank, False otherwise.
        """
        extrema = image.convert("L").getextrema()
        return extrema[0] == extrema[1]

    def initialize_image_ai(self, history):
        """
        Initialize the generative AI model for image-based interactions.

        Args:
            history (list): List of chat history.

        Returns:
            GenerativeModel: Initialized generative model for image chat interactions.
        """
        try:
            vision_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
                safety_settings=safety_settings,
                generation_config=generation_config,
                system_instruction=image_retriever_instruction,
            )
            return vision_model.start_chat(history=history)
        except Exception as e:
            print(e)
            return None

    def background_upload(self):
        """
        Uploads images to Gemini in the background for later processing.

        Returns:
            None
        """
        try:
            geminiUtils = GeminiUtils()
            uploaded_images = geminiUtils.upload_all_images(
                pdf_names=self.pdf_names)

        except Exception as e:
            print(e)
            return None

    def get_relevant_text(self, pdf_base_name, relevant_page_numbers):
        """
        Extracts text from specific pages of a PDF based on provided page numbers.

        Args:
            pdf_base_name (str): Base name of the PDF file.
            relevant_page_numbers (list): List of page numbers to extract text from.

        Returns:
            list: List of text extracted from the specified pages.
        """
        try:
            relevant_text = []
            print(f"Getting relevant text for {pdf_base_name}")
            print(f"Relevant page numbers: {relevant_page_numbers}")
            for pdf, pdf_name in zip(self.pdf_docs, self.pdf_names):
                print(f"Processing {pdf_name} and {pdf_base_name}")

                if (pdf_name == f"{pdf_base_name}.pdf"):
                    pdf_reader = PdfReader(pdf)
                    for page_num, page in enumerate(pdf_reader.pages, start=0):
                        print(f"Processing page {page_num}")
                        if page_num in relevant_page_numbers:
                            text = page.extract_text()
                            relevant_text.append(text)

            return relevant_text
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {e}")
