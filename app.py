import streamlit as st
import cv2
import os
import subprocess

# Directory paths
TEMPLATES_DIR = 'templates/'
OUTPUT_DIR = 'output/'

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def add_wrapped_text_to_frame(frame, text, position, font_scale, font_color, font, thickness=3):
    """Add wrapped text to a single frame."""
    height, width, _ = frame.shape
    lines = []
    words = text.split(' ')
    current_line = words[0]
    for word in words[1:]:
        if cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0][0] <= width - 20:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    line_height = cv2.getTextSize('Test', font, font_scale, thickness)[0][1] + 10
    if position == 'top':
        y = line_height + 40
    elif position == 'bottom':
        y = height - (len(lines) * line_height) - 40
    else:
        y = (height - (len(lines) * line_height)) // 2
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, line, (text_x, y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
        y += line_height
    return frame

def add_watermark(frame, text, position, font_scale, font_color, font, thickness=2):
    height, width, _ = frame.shape
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    if position == 'left':
        x = 10
    elif position == 'right':
        x = width - text_size[0] - 10
    else:
        x = (width - text_size[0]) // 2
    y = height - text_size[1] - 10
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
    return frame

def process_video(template_path, text, position, font_scale, font_color, font, output_path, target_fps, watermark_text=None, watermark_position='center', enhance_quality=False):
    # Extract audio using ffmpeg
    temp_audio_path = 'temp_audio.aac'
    subprocess.run(['ffmpeg', '-i', template_path, '-vn', '-acodec', 'copy', temp_audio_path], check=True)

    cap = cv2.VideoCapture(template_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.write(f"Video properties: Original FPS={original_fps}, Width={width}, Height={height}, Target FPS={target_fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    processed_frame_count = 0
    first_frame_shown = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if enhance_quality:
            frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)

        frame = add_wrapped_text_to_frame(frame, text, position, font_scale, font_color, font)
        if watermark_text:
            frame = add_watermark(frame, watermark_text, watermark_position, font_scale, font_color, font)

        out.write(frame)
        processed_frame_count += 1

        if not first_frame_shown:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Sample Frame')
            first_frame_shown = True

    cap.release()
    out.release()

    # Merge video and audio using ffmpeg
    subprocess.run(['ffmpeg', '-i', output_path, '-i', temp_audio_path, '-c', 'copy', 'final_output.mp4'], check=True)

    os.remove(output_path)
    os.remove(temp_audio_path)

    st.info(f"Processed {processed_frame_count} frames.")
    st.info(f"Output video saved at final_output.mp4")

    if os.path.exists('final_output.mp4'):
        st.success(f"Video successfully created at final_output.mp4")
        st.markdown(f"[Download Video](final_output.mp4)", unsafe_allow_html=True)
    else:
        st.error("Failed to create the output video.")

def create_video_from_text(text, template_path, position, font_scale, font_color, font, target_fps, watermark_text=None, watermark_position='center', enhance_quality=False):
    output_video_path = os.path.join(OUTPUT_DIR, 'output_video.mp4')
    process_video(template_path, text, position, font_scale, font_color, font, output_video_path, target_fps, watermark_text, watermark_position, enhance_quality)
    return output_video_path

# Streamlit app
st.title("Meme Video Generator")

text_input = st.text_input("Enter the meme text:")
template_source = st.selectbox("Template source:", ["Choose from existing", "Upload your own"])

if template_source == "Choose from existing":
    template_name = st.selectbox("Choose a template:", os.listdir(TEMPLATES_DIR))
    template_path = os.path.join(TEMPLATES_DIR, template_name)
else:
    uploaded_file = st.file_uploader("Upload a video template", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        template_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(template_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded file saved as {template_path}")

position = st.selectbox("Choose text position:", ["top", "center", "bottom"])
font_scale = st.slider("Select text size (scale):", 1.0, 10.0, 2.0)
font_color_hex = st.color_picker("Select text color:", "#FFFFFF")
font_color_bgr = tuple(int(font_color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

font_options = {
    "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "Duplex": cv2.FONT_HERSHEY_DUPLEX,
    "Complex": cv2.FONT_HERSHEY_COMPLEX,
    "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "Complex Small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "Script Simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "Script Complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX
}
font_choice = st.selectbox("Choose font type:", list(font_options.keys()))
font = font_options[font_choice]

target_fps = st.slider("Select target FPS:", 1, 240, 30)

watermark_text = st.text_input("Enter watermark text:")
watermark_position = st.selectbox("Choose watermark position:", ["left", "center", "right"])

enhance_quality = st.checkbox("Enhance video quality")

if st.button("Generate Video"):
    if text_input and template_path:
        video_path = create_video_from_text(text_input, template_path, position, font_scale, font_color_bgr, font, target_fps, watermark_text, watermark_position, enhance_quality)
    else:
        st.error("Please provide all inputs.")
