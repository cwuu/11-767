# reference: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
import math
import numpy as np
from model import SeeInDark, Light_SeeInDark, Lighter_SeeInDark
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import torch
import glob
import os 
import rawpy
import cv2
from PIL import Image

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_mse(img1, img2):
    return mean_squared_error(img1, img2)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = np.maximum(raw - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def evaluate(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    #get train and test IDs
    train_fns = glob.glob(args['gt_dir'] + '0*.ARW')
    train_ids = []
    for i in range(len(train_fns)):
        _, train_fn = os.path.split(train_fns[i])
        train_ids.append(int(train_fn[0:5]))

    test_fns = glob.glob(args['gt_dir']+ '/1*.ARW')
    test_ids = []
    for i in range(len(test_fns)):
        _, test_fn = os.path.split(test_fns[i])
        test_ids.append(int(test_fn[0:5]))

    if args['arch'] == "base":
        model = SeeInDark()
    elif args['arch'] == "light":
        model = Light_SeeInDark()
    elif args['arch'] == 'lighter':
        model = Lighter_SeeInDark()
    else:
        raise NotImplementedError # add your quantize model here
    model_path = args['model_dir'] + "_%s" % (args['arch']) + args['model_name']
    print(model_path)
    model.load_state_dict(torch.load(model_path ,map_location={'cuda:1':'cuda:0'}))
    model = model.to(device)

    if args['test_set']:
        dataset = test_ids 
        result_dir = os.path.join("eval", args['result_dir']+"_%s" % (args['arch']), "test", args['arch'])
    else:
        dataset = train_ids 
        result_dir = os.path.join("eval", args['result_dir']+"_%s" % (args['arch']), "train", args['arch'])
    print(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    avg_ssim, avg_psnr, avg_mse = [], [], []
    for test_id in dataset:
        #test the first image in each sequence
        in_files = glob.glob(args['input_dir'] + '%05d_00*.ARW'%test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(args['gt_dir'] + '%05d_00*.ARW'%test_id) 
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure =  float(in_fn[9:-5])
            gt_exposure =  float(gt_fn[9:-5])
            ratio = min(gt_exposure/in_exposure,300)

            raw = rawpy.imread(in_path)
            im = raw.raw_image_visible.astype(np.float32) 
            input_full = np.expand_dims(pack_raw(im),axis=0) *ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)	

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

            input_full = np.minimum(input_full,1.0)

            in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
            out_img = model(in_img)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()

            output = np.minimum(np.maximum(output,0),1)

            output = output[0,:,:,:]
            gt_full = gt_full[0,:,:,:]
            scale_full = scale_full[0,:,:,:]
            #origin_full = scale_full
            #scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full) # scale the low-light image to the same mean of the groundtruth
            cur_mse = calculate_mse(gt_full*255, output*255) / 255.
            cur_ssim = calculate_ssim(gt_full*255, output*255)
            cur_psnr = calculate_psnr(gt_full*255, output*255)
            print(("ssim:%f psnr:%f mse:%f" % (cur_ssim, cur_psnr, cur_mse)))
            avg_ssim.append(cur_ssim)
            avg_psnr.append(cur_psnr)
            avg_mse.append(cur_mse)

            if args['generate_img']:
                #Image.fromarray((origin_full*255).astype('uint8')).save(result_dir + '/%5d_00_%d_ori.png'%(test_id,ratio))
                Image.fromarray((output*255).astype('uint8')).save(result_dir + '/%5d_00_%d_out.png'%(test_id,ratio))
                #Image.fromarray((scale_full*255).astype('uint8')).save(result_dir + '%5d_00_%d_scale.png'%(test_id,ratio))
                Image.fromarray((gt_full*255).astype('uint8')).save(result_dir + '/%5d_00_%d_gt.png'%(test_id,ratio))
        avg_ssim = np.array(avg_ssim).mean()
        avg_psnr = np.array(avg_psnr).mean()
        avg_mse = np.array(avg_mse).mean()
        f = open(os.path.join(result_dir, 'result.txt'), 'a')
        f.write('\n')
        f.write("%f %f %f" % (avg_ssim, avg_psnr, avg_mse))
        f.close()



args = {
    "input_dir": './dataset/Sony/short/',
    "gt_dir": './dataset/Sony/long/',
    "result_dir": './result_Sony',
    "model_dir": './saved_model',
    "model_name": '/checkpoint_sony_e4000.pth',
    "arch": "light", #base, light, lighter
    "ps": 512, #patch size for training
    "save_freq":100,
    "total_epoch":4001,
    "generate_img": True,
    "test_set": True
}


if __name__ == '__main__':
    evaluate(args)
