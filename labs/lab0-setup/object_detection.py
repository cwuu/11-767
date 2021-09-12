import torch
import torchvision.models as models
import cv2, os, sys
import torchvision.transforms as transforms
from torchvision.utils import make_grid

#ref: https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.py
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
if __name__ == "__main__":
    cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    model = models.mobilenet_v2(pretrained=True)
    model.cuda()
    model.eval()
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((320, 320)), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #ref: https://pytorch.org/vision/stable/models.html

    
    x = 0
    if cam.isOpened():
        #Warm up
        while True:
            val, img = cam.read()
            cv2.imwrite("testing%d.png"%(x), img)
            img = data_transform(img).unsqueeze(0).cuda()
            out = model(img)
            print(out)
    else:
        print("Camera is not connected!")
