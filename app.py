import streamlit as st
import pytesseract
from PIL import Image
import os
from difflib import SequenceMatcher
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from google.cloud import vision
from google.auth.exceptions import DefaultCredentialsError
from io import BytesIO
from langdetect import detect, DetectorFactory

# Ensure uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Set up Streamlit app
st.set_page_config(page_title="OCR Text Extractor", page_icon="🔒", layout="centered")

# Header
st.title("OCR Text Extractor")

# OCR service selection
ocr_service = st.selectbox("Choose OCR Service", ["Tesseract", "Azure", "Google"])

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# File uploader for ground truth text
uploaded_text = st.file_uploader("Upload ground truth text file (optional)...", type=["txt"])

# Azure OCR setup
azure_subscription_key = st.text_input("Enter Azure Subscription Key (optional)")
azure_endpoint = st.text_input("Enter Azure Endpoint (optional)")
if azure_subscription_key and azure_endpoint:
    computervision_client = ComputerVisionClient(azure_endpoint, CognitiveServicesCredentials(azure_subscription_key))
else:
    computervision_client = None

# Google Cloud Vision setup
google_client = None
try:
    google_client = vision.ImageAnnotatorClient()
except DefaultCredentialsError:
    google_client = None

# Set seed for consistent language detection results
DetectorFactory.seed = 0

def calculate_similarity(ocr_text, ground_truth_text):
    similarity = SequenceMatcher(None, ocr_text, ground_truth_text).ratio()
    return similarity

def get_tesseract_text(image, lang):
    return pytesseract.image_to_string(image, lang=lang)

def get_azure_text(image_stream):
    if not computervision_client:
        return "Azure credentials are not set."
    ocr_result = computervision_client.read_in_stream(image_stream, raw=True)
    operation_location = ocr_result.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    # Polling for the OCR result
    while True:
        result = computervision_client.get_read_result(operation_id)
        if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
            break

    # Extracting text from the OCR result
    text = ""
    if result.status == OperationStatusCodes.succeeded:
        for read_result in result.analyze_result.read_results:
            for line in read_result.lines:
                text += line.text + "\n"
    return text

def get_google_text(image_stream):
    if not google_client:
        return "Google Cloud Vision API credentials are not set."
    content = image_stream.read()
    image = vision.Image(content=content)
    response = google_client.text_detection(image=image)
    texts = response.text_annotations
    return "\n".join([text.description for text in texts])

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def map_language_code(detected_language):
    if detected_language == "en":
        return "eng"
    elif detected_language == "hi":
        return "hin"
    else:
        return "eng"  # default to English if unknown

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Convert image to byte stream
    image_stream = BytesIO()
    image.save(image_stream, format='PNG')
    image_stream.seek(0)

    st.write("Extracted Text:")
    with st.spinner('Processing...'):
        try:
            ocr_text = ""
            if ocr_service == "Tesseract":
                detected_language = detect_language(pytesseract.image_to_string(image, lang='hin+eng'))
                lang_code = map_language_code(detected_language)
                ocr_text = get_tesseract_text(image, lang=lang_code)
            elif ocr_service == "Azure":
                ocr_text = get_azure_text(image_stream)
            elif ocr_service == "Google":
                ocr_text = get_google_text(image_stream)

            st.text_area("OCR Result", ocr_text, height=200)

            if uploaded_text is not None:
                # Read the ground truth text
                ground_truth_text = uploaded_text.read().decode('utf-8')
                st.text_area("Ground Truth Text", ground_truth_text, height=200)

                # Calculate accuracy
                similarity = calculate_similarity(ocr_text, ground_truth_text)
                st.write(f"Accuracy: {similarity * 100:.2f}%")
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem;">
        <small>Developed by <a href="https://www.linkedin.com/in/your-profile" target="_blank" style="color: #4CAF50;">Lakshay Madaan</a></small>
    </div>
    """,
    unsafe_allow_html=True,
)
