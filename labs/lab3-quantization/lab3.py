import numpy as np
import torchvision.models as models
import torch 
from timeit import default_timer as timer
from PIL import Image
from torchvision import transforms, datasets
from torch import nn
import torch.quantization
from unet import *
import os

#reference: https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

def run_benchmark(lab2_args):
    #unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #in_channels=3, out_channels=1, init_features=32, pretrained=lab2_args['pretrained'])



    in_channels  = 3
    out_channels = 5
    layout_encoder = [[[32,5,2,2]],
                     [[32,3,1,1],[32,3,2,1]],
                     [[64,3,1,1],[64,3,2,1]],
                     [[128,3,1,1],[128,3,2,1]]]

    model_zoo_org = {}
    model_zoo_org["unet"] = UNET_QAT(get_encoder(in_channels, layout_encoder), out_channels)
    model_zoo_org["sufflenet"] = models.shufflenet_v2_x1_0(pretrained=lab2_args['pretrained'])
    model_zoo_org["mobilenet_v3_small"] = models.mobilenet_v3_small(pretrained=lab2_args['pretrained'])
    model_zoo_org["resnet18"] =  models.resnet18(pretrained=lab2_args['pretrained'])

    unet = UNET_QAT(get_encoder(in_channels, layout_encoder), out_channels)
    unet = unet.eval()
    sufflenet = models.shufflenet_v2_x1_0(pretrained=lab2_args['pretrained'])
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=lab2_args['pretrained'])
    resnet18 = models.resnet18(pretrained=lab2_args['pretrained'])
    
    #Convert the quantization model
    torch.backends.quantized.engine = 'qnnpack'
    unet.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine) 
    unet = torch.quantization.prepare(unet, inplace=True)
    torch.quantization.convert(unet, inplace=True)
    sufflenet = torch.quantization.quantize_dynamic(sufflenet, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    mobilenet_v3_small = torch.quantization.quantize_dynamic(mobilenet_v3_small, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    resnet18 = torch.quantization.quantize_dynamic(resnet18, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    
    model_zoo = {}
    model_zoo["unet"] = unet 
    model_zoo["sufflenet"] = sufflenet
    model_zoo["mobilenet_v3_small"] = mobilenet_v3_small
    model_zoo["resnet18"] = resnet18
    #device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    device = "cpu"

    test_transforms = transforms.Compose([transforms.Resize(lab2_args['input_size']), transforms.ToTensor(),])
    test_data = datasets.ImageFolder(lab2_args['data_dir'], transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=lab2_args['batch_size'])

    #Save the quantized model

    for k, v in model_zoo.items():
        print("====%s====="%(k))
	# compare the sizes
        f=print_size_of_model(model_zoo_org[k],"fp32")
        q=print_size_of_model(model_zoo[k],"int8")
        print("{0:.2f} times smaller".format(f/q))
    assert(0)
    f = open('./r.txt', 'w+') 
	
    for k, v in model_zoo.items():
        print(" ========= Evaluating: {} =========".format(k))
        f.write(" ========= Evaluating: {} =========\n".format(k))
        model = v
        # latency (inference) Time in seconds
        start = timer()
        model.to(device)
        model.eval()
        criterion = nn.NLLLoss()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                if k == "unet":
                    output = model(inputs)
                else:
                    output = model.forward(inputs)
                break
                #test_loss += batch_loss.item()
                #ps = torch.exp(logps)
                #top_p, top_class = ps.topk(1, dim=1)
                #equals = top_class == labels.view(*top_class.shape)
                #accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # inference based on args setting
        end = timer()
        f.write(" Latency(input_size {}, batch_size {}): {} \n".format(lab2_args['input_size'], lab2_args['batch_size'], (end - start)))
        print(" Latency(input_size {}, batch_size {}): {} ".format(lab2_args['input_size'], lab2_args['batch_size'], (end - start)))
        


    print("====== Completed All Model Evaluation ======")
    f.write("====== Completed All Model Evaluation ======\n")
    f.close()


    
lab2_args = {
    "input_size":224, #256, 512, 1024
    "batch_size":4,
    "pretrained":True,
    "data_dir": "../tiny-imagenet-200/test/"

}
if __name__ == "__main__":
    input_sizes = [224, 448]#, 896]
    batch_sizes = [1, 2, 4, 8, 16, 32]#] 16, 32]
    for i in input_sizes:
        for b in batch_sizes:
            lab2_args["input_size"] = i
            lab2_args["batch_size"] = b
            run_benchmark(lab2_args)
