# Florence-2 Vision Model v1
This repository contains code for deploying an image processing application using the Florence-2-large model from Hugging Face(app.py) and gradio app (gradio_app.py), integrated with Gradio for a user-friendly interface.

### Overview
The Florence-2 Vision Model v1 leverages state-of-the-art natural language processing and computer vision capabilities to perform various tasks on uploaded images. These tasks include:

- Image Captioning: Generating captions for images.
- Object Detection: Identifying and localizing objects in images.
- OCR (Optical Character Recognition): Extracting text from images.
- Phrase Grounding: Associating phrases from captions with specific regions in the image.
- Cascaded Tasks: Performing detailed captioning and phrase grounding sequentially.
The application uses the `microsoft/Florence-2-large` model, fine-tuned and optimized for these tasks.

### Features
- Upload Images: Users can upload images through the web interface.
- Real-time Processing: Immediate feedback and results visualization for uploaded images.
- Multiple Tasks: Simultaneously performs multiple vision-related tasks on the same image.
- Visualization: Outputs include annotated images showing detected objects and phrases.

### Installation
To run this application locally or deploy it yourself, follow these steps:

- Clone the repository:
`git clone https://github.com/arad1367/florence-2-vision-model-v1.git`
`cd florence-2-vision-model-v1`

### Install dependencies:
`pip install -r requirements.txt`
`Note: Ensure that all dependencies, including gradio, transformers, and torch, are compatible and properly installed. You may need to adjust versions based on your environment.`

`Important note about flash_attn`
- I had some error to install this library, check original code and check below link if you have some problems:
`https://pypi.org/project/flash-attn/0.2.4/`

### Run the application:
`python app.py`
This command will start a local server. Open a web browser and go to `http://localhost:7860` to access the application (local). If you deployed on huggingface just use original code in `app.py` and not `app_gradio.py`

### Usage
Upload an image using the provided interface.
Wait for the application to process the image. Multiple tasks will be performed in parallel.
View the results for each task, including image captions, detected objects, OCR text, and annotated images showing phrase grounding.

### Deployment
This application can also be deployed on platforms like Hugging Face Spaces for broader accessibility. To deploy on Hugging Face, follow these steps:

Push your repository to GitHub (e.g., arad1367/florence-2-vision-model-v1).
Create a new repository on Hugging Face Spaces.
Connect your GitHub repository to Hugging Face Spaces and deploy the model.
Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

Fork the repository and create your branch from main.
Make your contributions, ensuring adherence to the project's coding style and guidelines.
Submit a pull request detailing your changes and improvements.

### Credits
- Pejman Ebrahimi - App development and integration with Florence-2-large model.
- Hugging Face Community - Models, transformers library, and deployment infrastructure.
- Florence-2-Microsoft model

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
For questions or inquiries, feel free to reach out to Pejman Ebrahimi:

- LinkedIn: `https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/`
- GitHub: `https://github.com/arad1367`
- email: `pejman.ebrahimi77@gmail.com`