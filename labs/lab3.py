import numpy as np
import torchvision.models as models
import torch 
from timeit import default_timer as timer
from PIL import Image
from torchvision import transforms
from torch import nn


def load_testloader(datadir, input_size=224, bath_size=4):
    test_transforms = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor(),])
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    test_idx = indices[:split]
    test_sampler = SubsetRandomSampler(test_idx)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return testloader

def run_benchmark(args):
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=args['pretrained'])
    squeezenet = models.squeezenet1_0(pretrained=args['pretrained'])
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=args['pretrained'])
    resnet18 = models.resnet18(pretrained=args['pretrained'])
    model_zoo = {}
    model_zoo["unet"] = unet 
    model_zoo["squeezenet"] = squeezenet
    model_zoo["mobilenet_v3_small"] = mobilenet_v3_small
    model_zoo["resnet18"] = resnet18
    #device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    device = "cpu"
    testloader = load_testloader(args['data_dir'], args['input_size']=224, args['batch_size'])
    for k, v in model_zoo.items():
        print(" ========= Evaluating: {} =========".format(k))
        model = v.copy()
        # parameter count
        num_params = sum([np.prod(p.size()) for p in v.parameters()])
        print(" 1. Param Count: {} ".format(num_params))
        # latency (inference) Time in seconds
        start = timer()
        model.to(device)
        model.eval()
        criterion = nn.NLLLoss()
        with torch.no_grad():
            for inputs, labels, in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    #test_loss += batch_loss.item()
                    #ps = torch.exp(logps)
                    #top_p, top_class = ps.topk(1, dim=1)
                    #equals = top_class == labels.view(*top_class.shape)
                    #accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # inference based on args setting
        end = timer()
        print(" 2. Latency(input_size {}, batch_size {}): {} ".format(args['input_size'], args['batch_size'], (end - start)))
        # energy use
    print("====== Completed All Model Evaluation ======")

args = {
    "input_size":60, #256, 512, 1024
    "batch_size":4,
    "pretrained":True,
    "data_dir": "./"

}
if __name__ == "__main__":
    run_benchmark(args)