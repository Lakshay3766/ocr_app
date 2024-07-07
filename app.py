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

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = os.path.join(os.path.dirname(__file__), 'tessdata')

# Ensure uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Set up Streamlit app
st.set_page_config(page_title="OCR Text Extractor", page_icon="🔒", layout="centered")

# Header
st.title("OCR Text Extractor")

# Language selection
language_option = st.selectbox("Choose OCR Language", ["Automatic Detection", "English", "Hindi"])

# OCR service selection
st.markdown("### Select OCR Services to Use:")
use_tesseract = st.checkbox("Tesseract", value=True)
use_azure = st.checkbox("Azure", value=False)
use_google = st.checkbox("Google", value=False)
use_combined = st.checkbox("Combined", value=False)

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# File uploader for ground truth text
uploaded_text = st.file_uploader("Upload ground truth text file (optional)...", type=["txt"])

# Azure OCR setup
azure_subscription_key = st.text_input("Enter Azure Subscription Key (optional)", type="password")
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

def calculate_similarity(text1, text2):
    similarity = SequenceMatcher(None, text1, text2).ratio()
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

def combine_ocr_results(results):
    """
    Combine OCR results from different services and select the most appropriate text.
    """
    # Simple voting mechanism: select the most common result
    from collections import Counter
    common_result = Counter(results).most_common(1)[0][0]
    
    return common_result

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
            if language_option == "Automatic Detection":
                detected_language = detect_language(pytesseract.image_to_string(image, lang='hin+eng'))
                lang_code = map_language_code(detected_language)
            elif language_option == "English":
                lang_code = "eng"
            elif language_option == "Hindi":
                lang_code = "hin"

            ocr_texts = []
            ocr_text = ""

            # OCR using Tesseract
            if use_tesseract:
                tesseract_text = get_tesseract_text(image, lang=lang_code)
                ocr_texts.append(tesseract_text)
                st.text_area("Tesseract OCR Result", tesseract_text, height=150)

            # OCR using Azure
            if use_azure:
                azure_text = get_azure_text(image_stream)
                ocr_texts.append(azure_text)
                st.text_area("Azure OCR Result", azure_text, height=150)
                image_stream.seek(0)  # Reset stream for next OCR

            # OCR using Google
            if use_google:
                google_text = get_google_text(image_stream)
                ocr_texts.append(google_text)
                st.text_area("Google OCR Result", google_text, height=150)
                image_stream.seek(0)  # Reset stream for next OCR

            # Combine results if the combined checkbox is selected
            if use_combined and ocr_texts:
                combined_result = combine_ocr_results(ocr_texts)
                st.text_area("Combined OCR Result", combined_result, height=150)

            if uploaded_text is not None:
                # Read the ground truth text
                ground_truth_text = uploaded_text.read().decode('utf-8')
                st.text_area("Ground Truth Text", ground_truth_text, height=200)

                # Calculate accuracy for each OCR result
                if use_tesseract:
                    tesseract_accuracy = calculate_similarity(tesseract_text, ground_truth_text)
                    st.write(f"Tesseract Accuracy: {tesseract_accuracy * 100:.2f}%")

                if use_azure:
                    azure_accuracy = calculate_similarity(azure_text, ground_truth_text)
                    st.write(f"Azure Accuracy: {azure_accuracy * 100:.2f}%")

                if use_google:
                    google_accuracy = calculate_similarity(google_text, ground_truth_text)
                    st.write(f"Google Accuracy: {google_accuracy * 100:.2f}%")

                if use_combined:
                    combined_accuracy = calculate_similarity(combined_result, ground_truth_text)
                    st.write(f"Combined Accuracy: {combined_accuracy * 100:.2f}%")

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
