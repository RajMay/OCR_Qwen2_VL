OCR Web Application using Qwen2-VL-2B-Instruct
This project is a web-based Optical Character Recognition (OCR) tool that uses the Qwen2-VL-2B-Instruct model for extracting text from images. The web interface is built using Streamlit, and the model is processed via the Hugging Face transformers library.

Features
Image Upload: Users can upload images in png, jpg, or jpeg formats.
OCR Functionality: The uploaded image is processed to extract text.
Keyword Search: Once the text is extracted, users can perform a keyword search within the extracted text, and matching words will be highlighted.
Requirements
To run this project locally, the following libraries are required:
pip install streamlit
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
pip install torch pillow accelerate
PROJECT STRUCTURE:
.
├── streamlit_app.py        # Main Streamlit application code
├── README.md               # This file
└── images/                 # Folder for storing sample images
How to Run
Clone the Repository:

bash
Copy code
git clone <repository-url>
cd <repository-folder>
Install the Required Libraries: Install the dependencies listed above using pip.

Run the Application: Start the Streamlit app by running the following command:


Copy code
streamlit run streamlit_app.py
Upload Image and Extract Text:

The application will open in your default browser.
Upload an image, and the OCR model will extract the text.
You can also use the keyword search functionality to find specific words within the extracted text.
Notes
The model used for OCR is loaded from Hugging Face's Qwen/Qwen2-VL-2B-Instruct. Ensure you have a good internet connection as the model is quite large.
This project was first developed in Google Colab for model testing, and later migrated to Streamlit for deployment.
