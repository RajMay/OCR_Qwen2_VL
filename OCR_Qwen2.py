import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# Display version information
print("Streamlit version:", st.__version__)
print("Torch version:", torch.__version__)
print("Pillow (PIL) version:", Image.__version__)

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map={"": "cpu"}
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Streamlit app title
st.title("OCR Image Text Extraction")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # OCR processing for text extraction
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract the text from this image."},
            ],
        }
    ]

    # Prepare input for the model
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=[text_input], padding=True, return_tensors="pt")
    inputs = inputs.to("cpu")

    # Generate text using the model
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Display extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Keyword search functionality
    st.subheader("Keyword Search")
    search_query = st.text_input("Enter keywords to search within the extracted text")

    if search_query:
        # Check if the search query is in the extracted text
        if search_query.lower() in extracted_text.lower():
            highlighted_text = extracted_text.replace(search_query, f"**{search_query}**")
            st.write(f"Matching Text: {highlighted_text}")
        else:
            st.write("No matching text found.")
