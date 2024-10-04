
from ultralytics import YOLO
import matplotlib.pyplot as plt 
from PIL import Image
import cv2
import os
import nltk
from scipy import spatial
import gensim.downloader as api
import numpy as np
from typing import List, Dict, Tuple, Any
import clip
import torch

####### helper functions

# helper function to resolve relative paths to absolute paths

def resolve_path(path):
    return os.path.abspath(os.path.expanduser(path))


# helper functions for finding synonyms to YOLO classes

word_vectors = api.load("glove-wiki-gigaword-100")

temp = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
class_list = list(temp.values())

def related_classes(word, class_list, threshold):
    # List to hold synonyms with their similarity score
    synonym_similarity = []

    # Calculate similarity for each class name in class_list
    for class_name in class_list:
        similarity = calculate_similarity(word, class_name)
        if (similarity > threshold):
            synonym_similarity.append((class_name, similarity))
    
    # Sort synonyms by similarity in descending order
    sorted_synonyms = sorted(synonym_similarity, key=lambda x: x[1], reverse=True)
    
    # Return the sorted list of class names (synonyms)
    return [syn for syn, sim in sorted_synonyms]

def calculate_similarity(word1, word2):
    # Try to get word vectors
    try:
        vec1 = word_vectors[word1]
        vec2 = word_vectors[word2]
        return 1 - spatial.distance.cosine(vec1, vec2)
    except KeyError:
        return 0  # If either word is not in the vocabulary, return 0 similarity


# helper visualization functions
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


# load input image

def load_image(image_path):
    image = Image.open(image_path)

    return image




# run YOLO object detection

def run_YOLO(image_path, YOLO_model):
    result = YOLO_model.predict(image_path)
    return result


# Iterate through YOLO results
def check_result(YOLO_result, target_class):
    cls_names = related_classes(target_class, class_list, 0.6)
    boxes = []
    for result_helmet in YOLO_result:
        names = result_helmet.names
        for box in result_helmet.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            name = names[cls]
            if name in cls_names:
                boxes.append([x1, y1, x2, y2])
            
    return boxes


# run box inference with SAM2
def box_sam2(boxes, predictor, image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    image_np = np.array(image)

    # Create a copy of the original image for overlaying masks
    overlay_image = image_np.copy()

    # Loop over each bounding box in boxes
    for box in boxes:
        # Convert the box to a numpy array and predict the mask using SAM
        input_box = np.array(box)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Extract the first mask and ensure it's binary (0 or 1)
        mask = masks[0].astype(np.uint8)

        # Create a red overlay where the mask is True
        red_mask = np.zeros_like(image_np)
        red_mask[mask == 1] = [255, 0, 0]  # Red color

        # Combine the current mask with the overlay image using alpha blending
        alpha = 0.5  # Transparency factor for the overlay
        overlay_image = cv2.addWeighted(overlay_image, 1, red_mask, alpha, 0)

    # Convert the result back to a PIL image and return
    return Image.fromarray(overlay_image)


# function to save image
def save_image(image, output_dir):
    # Ensure the directory exists
    output_dir = resolve_path(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    


    # Construct the full output path (filename can be customized as needed)
    output_path = os.path.join(output_dir, "output_image.png")
    
    # Save the image to the specified directory
    image.save(output_path)

    print(f"Image saved to {output_path}")


# function to find visual centers
# First finding the visual centers of all the masks

np.random.seed(3)
def get_mask_centers_and_areas(anns, visualize=False, method='bbox'):
    if len(anns) == 0:
        return []

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    centers_and_areas = []

    height, width = sorted_anns[0]['segmentation'].shape
    total_image_area = height * width

    if visualize:
        img = np.zeros((height, width, 4), dtype=np.float32)

    for ann in sorted_anns:
        m = ann['segmentation']
        
        # Calculate area
        area = np.sum(m)
        area_percentage = (area / total_image_area) * 100

        # Calculate center based on chosen method
        if method == 'bbox':
            # Bounding box center
            y, x = np.where(m)
            cX = int((x.min() + x.max()) / 2)
            cY = int((y.min() + y.max()) / 2)
        elif method == 'median':
            # Median point
            y, x = np.where(m)
            cX = int(np.median(x))
            cY = int(np.median(y))
        else:
            # Fallback to center of mass
            M = cv2.moments(m.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

        # Ensure the center is inside the mask
        if not m[cY, cX]:
            # If center is outside, find the nearest point inside the mask
            dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
            cY, cX = np.unravel_index(dist.argmax(), dist.shape)

        centers_and_areas.append({
            'center': (cX, cY),
            'area': area,
            'area_percentage': area_percentage
        })

        if visualize:
            color = np.concatenate([np.random.random(3), [0.5]])
            colored_mask = np.zeros((height, width, 4), dtype=np.float32)
            colored_mask[m] = color
            img = np.where(colored_mask > 0, colored_mask, img)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 0, 1), thickness=1)
            cv2.circle(img, (cX, cY), 5, (1, 0, 0, 1), -1)

    if visualize:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return centers_and_areas



def process_centers_with_sam2(
    centers_and_areas: List[Dict[str, Any]],
    predictor: Any,
    image_shape: Tuple[int, int],
    granularity: float = 100.0
) -> Dict[Tuple[int, int], List[Tuple[np.ndarray, float]]]:
    """
    Process image centers using SAM2 point-based inferencing, sorted by area.

    Args:
    centers_and_areas (List[Dict]): List of dictionaries containing center coordinates and areas.
    predictor (SAM2ImagePredictor): An instance of SAM2ImagePredictor.
    image_shape (Tuple[int, int]): Shape of the image (height, width).
    granularity (float): Percentage of image area to process (0-100).

    Returns:
    Dict[Tuple[int, int], List[Tuple[np.ndarray, float]]]: Dictionary with points as keys and list of (mask, score) as values.
    """
    result = {}
    total_image_area = image_shape[0] * image_shape[1]
    processed_area = 0
    target_area = (granularity / 100) * total_image_area

    # Sort centers_and_areas by area in descending order
    sorted_centers_and_areas = sorted(centers_and_areas, key=lambda x: x['area'], reverse=True)

    for item in sorted_centers_and_areas:
        center = item['center']
        area = item['area']

        # Check if we've processed enough area
        if processed_area >= target_area:
            break

        # Perform point-based inferencing
        input_point = np.array([center])
        input_label = np.array([1])  # Assuming 1 for positive point

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Sort masks by score
        sorted_indices = np.argsort(scores)[::-1]
        masks = masks[sorted_indices]
        scores = scores[sorted_indices]

        # Add results to the dictionary
        result[center] = list(zip(masks, scores))

        # Update processed area
        processed_area += area

    return result



def create_noun_vector_store(word_limit=1000):
    """
    currently made to fit the available test cases. Can be expanded.
    """
    print("Loading word vectors...")
#     word_vectors = api.load("glove-wiki-gigaword-100")
    unconditional_words = ['laptop', 'vase', 'lamp', 'table', 'sofa', 'chair']
#     print("Filtering for common nouns...")
    # Filter out proper nouns, symbols, and invalid words
#     nouns = [word for word in word_vectors.key_to_index.keys() if is_common_noun(word)]
    
    # Get the most common nouns (up to the limit)
#     common_nouns = nouns[:word_limit]
    common_nouns = []
    
    # Create a dictionary of noun vectors
    noun_vectors = {}
    
    # Handle unconditional word additions (no filtering)
    for word in unconditional_words:
        if word in word_vectors:  # Add the word directly if it exists in the word vectors
            common_nouns.append(word)
            noun_vectors[word] = word_vectors[word]
    
    print(f"Created noun vector store with {len(common_nouns)} nouns.")
    return common_nouns, noun_vectors


def get_top_words(image, model, preprocess, noun_store, num_words=3):
    common_nouns, noun_vectors = noun_store
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert NumPy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Tokenize nouns for CLIP
    text_inputs = clip.tokenize(["This is " + noun for noun in common_nouns]).to(device)
    print("Gave input to CLIP!")

    # Get the image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    print("Finding similarity!")
    # Calculate the similarity between the image and each noun
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    print("Finding top K words!")
    # Get the top N nouns
    values, indices = similarity[0].topk(num_words)

    return [(common_nouns[idx], val.item()) for val, idx in zip(values, indices)]


def process_masks(image, masks, scores, model, preprocess, noun_store):
    output_list = []  # To store the (image_with_red_mask, top_words) tuples
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Convert PIL image to NumPy array
        img = np.array(image)
        m = mask.astype(bool)
        
        # Create a blank white image (for the mask overlay)
        masked_image = np.zeros_like(img) + 255
        masked_image[m] = img[m]  # Apply the mask to the image
        
        # Ensure the masked image has 3 channels (RGB)
        if masked_image.shape[-1] != 3:
            masked_image = np.stack((masked_image,) * 3, axis=-1)
        
        # Do CLIP inferencing
        top_words = str(get_top_words(masked_image, model, preprocess, noun_store, num_words=3))
        
        # Create a red mask for the overlay
        red_mask = np.zeros_like(img)
        red_mask[m] = [255, 0, 0]  # Red color for the mask
        
        # Overlay the red mask onto the original image
        alpha = 0.5  # Transparency for the red mask
        overlay_image = cv2.addWeighted(img, 1, red_mask, alpha, 0)
        
        # Convert the overlay image back to PIL format
        overlay_image_pil = Image.fromarray(overlay_image)
        
        # Append the result (overlay image + top_words) to the list
        output_list.append((overlay_image_pil, top_words, score))
    
    # Return the list of (image_with_red_mask, top_words) tuples
    output_list.sort(key= lambda x: x[2], reversed=True)
    return output_list



# function to save all potential matching masked images
import os

def save_matching(processed_list, target_class, output_dir='output_images'):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Counter to generate unique filenames
    image_count = 0

    # Iterate through the processed list
    for i, (image, top_words, score) in enumerate(processed_list):
        # Filter based on score > 0.2
        if score > 0.2:
            # Split top_words to get individual words and their confidence scores
            top_words_list = top_words.strip('[]').split(', ')
            
            # Iterate over the top words and check each one
            for word_score in top_words_list:
                word, confidence = word_score.rsplit(':', 1)  # Assuming "word: score" format
                confidence = float(confidence)
                
                # Check if the target_class is present and has confidence >= 0.5
                if word == target_class and confidence >= 0.5:
                    # Save the image
                    output_path = os.path.join(output_dir, f"masked_image_{image_count}.png")
                    image.save(output_path)
                    print(f"Saved {output_path} with score: {score} and confidence: {confidence}")
                    image_count += 1
