import argparse
from pipeline import process_image
from pipeline import resolve_path, load_image, run_YOLO, check_result, box_sam2, save_image, get_mask_centers_and_areas, process_centers_with_sam2, create_noun_vector_store, process_masks,save_matching
from ultralytics import YOLO
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Object segmentation tool")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--target_class", required=True, help="Class to segment")
    parser.add_argument("--output", required=True, help="Path to output image")
    args = parser.parse_args()
    
    image_path = args.image

    target_class = args.target_class

    output_dir = args.output

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    

    YOLO_model = YOLO(resolve_path('./checkpoints/yolov9e.pt'))

    target_image = load_image(image_path=image_path)

    YOLO_result = run_YOLO(image_path=image_path, YOLO_model=YOLO_model)

    boxes = check_result(YOLO_result, target_class)
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, resolve_path(sam2_checkpoint), device=device)
    if (len(boxes) > 0):
        predictor = SAM2ImagePredictor(sam2_model)

        image = Image.open(image_path)
        predictor.set_image(image)

        final_image = box_sam2(boxes, predictor, image_path)
        save_image(final_image, output_dir)

    else:
        mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

        masks = mask_generator.generate(image)

        centers_and_areas = get_mask_centers_and_areas(masks, visualize=False, method="median")

        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        result = process_centers_with_sam2(centers_and_areas, predictor, image.shape[:2], granularity=80.0)

        model, preprocess = clip.load("ViT-L/14@336px", device="cuda" if torch.cuda.is_available() else "cpu")

        noun_store = create_noun_vector_store()

        for input_point, ms in result.items():
            masks = []
            scores = []
            for m, s in ms:
                masks.append(m)
                scores.append(s)
            
            processed_list = process_masks(image, masks, scores, model, preprocess, noun_store)

            save_matching(processed_list, target_class, output_dir)
    
    return

if __name__ == "__main__":
    main()