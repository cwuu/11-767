import torch
import torchvision.models as models
import cv2, os, sys
from torchvision.utils import make_grid

#HEIGHT=1280
#WIDTH=1920

def gstreamer_pipeline(capture_width=1280, capture_height=720, 
                       display_width=1280, display_height=720,
                       framerate=60, flip_method=0):
  return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

if __name__ == "__main__":
    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER) #Get camera 
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    model.cuda()
    model.eval()
    data_transform = transforms.Compose([transforms.Resize((320, 320)), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.ToTensor()]) #https://pytorch.org/vision/stable/models.html
    
    if cam.isOpened():
        while True:
            val, img = cam.read()
            img = data_transform(img).unsqueeze(0).cuda()
            out = model(img)
            print(out)
    else:
        print("Camera is not connected!")
    