import kagglehub
import cv2

# Download latest version
path = kagglehub.dataset_download("bestofbests9/icdar2015")

print("Path to dataset files:", path)
