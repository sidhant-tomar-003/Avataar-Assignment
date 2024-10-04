import subprocess
import sys
import os
import requests

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode())
        sys.exit(1)
    return stdout.decode()

def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def main():
    # Create checkpoints directory
    os.makedirs("./checkpoints", exist_ok=True)

    # Install requirements
    print("Installing requirements...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")

    # Download YOLOv9 model
    print("Downloading YOLOv9 model...")
    yolo_url = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9e.pt"
    yolo_path = "./checkpoints/yolov9e.pt"
    download_file(yolo_url, yolo_path)
    print(f"YOLOv9 model downloaded to {yolo_path}")

    # Download SAM2 model
    print("Downloading SAM2 model...")
    run_command("wget -P ./checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")

    # Download CLIP model
    print("Downloading CLIP model...")
    run_command(f"{sys.executable} -c \"import clip; clip.load('ViT-L/14@336px')\"")

    print("Initialization complete!")

if __name__ == "__main__":
    main()