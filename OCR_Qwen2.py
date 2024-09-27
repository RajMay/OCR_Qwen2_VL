import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

import streamlit
import torch
import transformers
import qwen_vl_utils
import PIL

print("Streamlit version:", streamlit.__version__)
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Pillow (PIL) version:", PIL.__version__)
##print("Qwen-vl-utils version:", qwen_vl_utils.__version__)



model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto" ,device_map={"": "cpu"}
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

st.title("OCR.....")




uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # OCR Processing
    image_path = uploaded_file.name
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Extract the text from this image."},
            ],
        }
    ]

    # Prepare inputs for the model
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    from accelerate import init_empty_weights


   
    

    
    inputs = inputs.to( "cpu")
    
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
  
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    st.subheader("Keyword Search")
    search_query = st.text_input("Enter keywords to search within the extracted text")

    if search_query:
       
        if search_query.lower() in extracted_text.lower():
            highlighted_text = extracted_text.replace(search_query, f"**{search_query}**")
            st.write(f"Matching Text: {highlighted_text}")
        else:
            st.write("No matching text found.")
