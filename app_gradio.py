import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import numpy as np

# Initialize Florence-2-large model and processor
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Function to resize and preprocess image
def preprocess_image(image_path, max_size=(800, 800)):
    image = Image.open(image_path).convert('RGB')
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.LANCZOS)

    # Convert image to numpy array
    image_np = np.array(image)

    # Ensure the image is in the format [height, width, channels]
    if image_np.ndim == 2:  # Grayscale image
        image_np = np.expand_dims(image_np, axis=-1)
    elif image_np.shape[0] == 3:  # Image in [channels, height, width] format
        image_np = np.transpose(image_np, (1, 2, 0))

    return image_np, image.size

# Function to run Florence-2-large model
def run_florence_model(image_np, image_size, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image_np, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=image_size
    )

    return parsed_answer, generated_text

# Function to plot image with bounding boxes
def plot_image_with_bboxes(image_np, bboxes, labels=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan']
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        x, y, width, height = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        if labels and i < len(labels):
            ax.text(x, y, labels[i], color=color, fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    plt.axis('off')
    return fig

# Gradio function to process uploaded images
def process_image(image_path):
    image_np, image_size = preprocess_image(image_path)

    # Image Captioning
    caption_result, _ = run_florence_model(image_np, image_size, '<CAPTION>')
    detailed_caption_result, _ = run_florence_model(image_np, image_size, '<DETAILED_CAPTION>')

    # Object Detection
    od_result, _ = run_florence_model(image_np, image_size, '<OD>')
    od_bboxes = od_result['<OD>'].get('bboxes', [])
    od_labels = od_result['<OD>'].get('labels', [])

    # OCR
    ocr_result, _ = run_florence_model(image_np, image_size, '<OCR>')

    # Phrase Grounding
    pg_result, _ = run_florence_model(image_np, image_size, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=caption_result['<CAPTION>'])
    pg_bboxes = pg_result['<CAPTION_TO_PHRASE_GROUNDING>'].get('bboxes', [])
    pg_labels = pg_result['<CAPTION_TO_PHRASE_GROUNDING>'].get('labels', [])

    # Cascaded Tasks (Detailed Caption + Phrase Grounding)
    cascaded_result, _ = run_florence_model(image_np, image_size, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=detailed_caption_result['<DETAILED_CAPTION>'])
    cascaded_bboxes = cascaded_result['<CAPTION_TO_PHRASE_GROUNDING>'].get('bboxes', [])
    cascaded_labels = cascaded_result['<CAPTION_TO_PHRASE_GROUNDING>'].get('labels', [])

    # Create plots
    od_fig = plot_image_with_bboxes(image_np, od_bboxes, od_labels)
    pg_fig = plot_image_with_bboxes(image_np, pg_bboxes, pg_labels)
    cascaded_fig = plot_image_with_bboxes(image_np, cascaded_bboxes, cascaded_labels)

    # Prepare response
    response = f"""
    Image Captioning:
    - Simple Caption: {caption_result['<CAPTION>']}
    - Detailed Caption: {detailed_caption_result['<DETAILED_CAPTION>']}

    Object Detection:
    - Detected {len(od_bboxes)} objects

    OCR:
    {ocr_result['<OCR>']}

    Phrase Grounding:
    - Grounded {len(pg_bboxes)} phrases from the simple caption

    Cascaded Tasks:
    - Grounded {len(cascaded_bboxes)} phrases from the detailed caption
    """

    return response, od_fig, pg_fig, cascaded_fig

# Gradio interface
with gr.Blocks(theme='NoCrypt/miku') as demo:
    gr.Markdown("""
    # Image Processing with Florence-2-large
    Upload an image to perform image captioning, object detection, OCR, phrase grounding, and cascaded tasks.
    """)

    image_input = gr.Image(type="filepath")
    text_output = gr.Textbox()
    plot_output_1 = gr.Plot()
    plot_output_2 = gr.Plot()
    plot_output_3 = gr.Plot()

    image_input.upload(process_image, inputs=[image_input], outputs=[text_output, plot_output_1, plot_output_2, plot_output_3])

    footer = """
    <div style="text-align: center; margin-top: 20px;">
        <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
        <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
        <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a>
        <br>
        Made with 💖 by Pejman Ebrahimi
    </div>
    """
    gr.HTML(footer)

demo.launch()