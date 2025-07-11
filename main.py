import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

import numpy as np
import argparse # Import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)


def run_example(task_prompt, text_input=None, pil_image=None):
    if text_input is None:    
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    # Prepare the inputs using the PIL image
    inputs = processor(text=prompt,images=pil_image,return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_image.width, pil_image.height))

    return parsed_answer


def plot_bbox(
    image,
    bboxes,
    labels=None,
    thickness=2,
    font_scale=0.8,
    text_color=(0, 0, 0),
    text_padding=5
):
    original_h, original_w, _ = image.shape
    
    dpi = 100
    fig, ax = plt.subplots(figsize=(original_w / dpi, original_h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image)

    for idx, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        width = x_max - x_min
        height = y_max - y_min
        color_rgb = (1, 0, 0)
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=thickness, edgecolor=color_rgb, facecolor='none'
        )
        ax.add_patch(rect)
        label_text = labels[idx] if labels and idx < len(labels) else f"Object {idx+1}"
        ax.text(
            x_min + text_padding, y_min - text_padding, label_text,
            fontsize=font_scale * 10, color='white',
            bbox=dict(facecolor=color_rgb, alpha=0.5, pad=text_padding)
        )

    ax.axis('off')

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Florence-2 tasks.")
    parser.add_argument('--task', type=str, required=True,
                        choices=['cap', 'od', 'rp', 'ov-od', 'cascaded'],
                        help="Specify the task to run.")
    parser.add_argument('--text_input', type=str,
                        help="Text input for tasks like Open-Vocabulary Object Detection.")
    parser.add_argument('--level', type=int, choices=[0, 1, 2],
                        help="level for cascaded tasks (0: caption, 1: phrase grounding, 2: dense captioning).")

    args = parser.parse_args()

    pil_image = Image.open('image.jpg')
    
    # Convert PIL image to RGB NumPy array for matplotlib
    numpy_image_rgb = np.array(pil_image.convert('RGB'))

    annotated_image = None
    text_to_display = None
    
    if args.task == 'cap':
        task_prompt = '<CAPTION>'
        result = run_example(task_prompt=task_prompt, pil_image=pil_image)
        text_to_display = result[task_prompt]
        fig, ax = plt.subplots(1)
        ax.imshow(numpy_image_rgb)
        ax.set_axis_off()

        if text_to_display:
            fig.text(0.5, 0.05, text_to_display, ha='center', fontsize=12, wrap=True)
            plt.subplots_adjust(bottom=0.2)

    elif args.task == 'od':
        task_prompt = '<OD>'
        result = run_example(task_prompt=task_prompt, pil_image=pil_image)
        bboxes = result['<OD>']['bboxes']
        labels = result['<OD>']['labels']
        plot_bbox(numpy_image_rgb, bboxes, labels)
        print(f"Object Detection Labels: {labels}")

    elif args.task == 'rp':
        task_prompt = '<REGION_PROPOSAL>'
        result = run_example(task_prompt=task_prompt, pil_image=pil_image)
        bboxes = result['<REGION_PROPOSAL>']['bboxes']

        labels = [f"Region {i+1}" for i in range(len(bboxes))] 
        plot_bbox(numpy_image_rgb, bboxes, labels)
        print(f"Region Proposal BBoxes: {bboxes}")

    elif args.task == 'ov-od':
        if not args.text_input:
            parser.error("--text_input is required for 'ov-od' task.")

        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        result = run_example(task_prompt=task_prompt, text_input=args.text_input, pil_image=pil_image)
        bboxes = result['<OPEN_VOCABULARY_DETECTION>']['bboxes']
        labels = result['<OPEN_VOCABULARY_DETECTION>']['bboxes_labels']
        plot_bbox(numpy_image_rgb, bboxes, labels)
        text_to_display = f"Query: {args.text_input}"
        print(f"Open-Vocabulary Detection Labels: {labels}")

    elif args.task == 'cascaded':

        if args.level == 0 | args.level == None:
            task_caption = '<CAPTION>'
        elif args.level == 1:
            task_caption = '<DETAILED_CAPTION>'
        elif args.level == 2:
            task_caption = '<MORE_DETAILED_CAPTION>'

        caption_result = run_example(task_prompt=task_caption, pil_image=pil_image)
        text_input_for_grounding = caption_result[task_caption]
        
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        result = run_example(task_prompt=task_prompt, text_input=text_input_for_grounding, pil_image=pil_image)
        bboxes = result['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
        labels = result['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
        plot_bbox(numpy_image_rgb, bboxes, labels)
        text_to_display = f"Caption: {text_input_for_grounding}"

    print(text_to_display)
    plt.show()