import numpy as np
import torchvision.models as models
import torch 
from timeit import default_timer as timer
from PIL import Image
from torchvision import transforms, datasets
from torch import nn


def run_benchmark(lab2_args):
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=lab2_args['pretrained'])
    squeezenet = models.squeezenet1_0(pretrained=lab2_args['pretrained'])
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=lab2_args['pretrained'])
    resnet18 = models.resnet18(pretrained=lab2_args['pretrained'])
    model_zoo = {}
    model_zoo["unet"] = unet 
    model_zoo["squeezenet"] = squeezenet
    model_zoo["mobilenet_v3_small"] = mobilenet_v3_small
    model_zoo["resnet18"] = resnet18
    #device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    device = "cpu"

    test_transforms = transforms.Compose([transforms.Resize(lab2_args['input_size']), transforms.ToTensor(),])
    test_data = datasets.ImageFolder(lab2_args['data_dir'], transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=lab2_args['batch_size'])

    for k, v in model_zoo.items():
        print(" ========= Evaluating: {} =========".format(k))
        model = v
        # parameter count
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        print(" 1. Param Count: {} ".format(num_params))
        # latency (inference) Time in seconds
        start = timer()
        model.to(device)
        model.eval()
        criterion = nn.NLLLoss()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                break
                #test_loss += batch_loss.item()
                #ps = torch.exp(logps)
                #top_p, top_class = ps.topk(1, dim=1)
                #equals = top_class == labels.view(*top_class.shape)
                #accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # inference based on args setting
        end = timer()
        print(" 2. Latency(input_size {}, batch_size {}): {} ".format(lab2_args['input_size'], lab2_args['batch_size'], (end - start)))
        # energy use
    print("====== Completed All Model Evaluation ======")

lab2_args = {
    "input_size":224, #256, 512, 1024
    "batch_size":4,
    "pretrained":True,
    "data_dir": "/home/nvidia/11767/lab2/tiny-imagenet-200/test/"

}
if __name__ == "__main__":
    input_sizes = [224, 448, 896]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    for i in input_sizes:
        for b in batch_sizes:
            lab2_args["input_size"] = i
            lab2_args["batch_size"] = b
            run_benchmark(lab2_args)
