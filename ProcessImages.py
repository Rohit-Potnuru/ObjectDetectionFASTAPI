import os
import time

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from PIL import Image, ImageDraw, ImageFont

from Config import *
from CustomUtils import *
import CustomUnpickler
from ModelArchitecture import *




"""
Process_Image_Object_Detect: input: List -> list of string notating image addresses
                             output: List -> list of string notating processed image addresses
"""

def processingImages(list_image, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):

    
    
    
    model = torch.load(MODEL_FILE, map_location=DEVICE, pickle_module=CustomUnpickler)['model']
    model.eval()
  
    output_imgs = []
    for img_path in list_image:
        image_file = os.path.join(input_dir, img_path)
        
        image = Image.open(image_file, mode='r')
        image = image.convert('RGB')

        output = detect(image, model, min_score=0.4, max_overlap=0.4, top_k=200, device=DEVICE)
        output_file = os.path.join(output_dir, "output_" + img_path)
        output.save(output_file)
        output_imgs.append("output_" + img_path)
    
    return output_imgs

def detect(original_image, model, min_score, max_overlap, top_k, suppress=None, device = "cpu"):
    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(DEVICE)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to(DEVICE)

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [REV_LABEL_MAP[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=LABEL_COLOR_MAP[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=LABEL_COLOR_MAP[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 2. for l in box_location], outline=LABEL_COLOR_MAP[
            det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 3. for l in box_location], outline=LABEL_COLOR_MAP[
            det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=LABEL_COLOR_MAP[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image