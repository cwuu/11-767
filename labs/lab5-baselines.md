Lab 4: Baselines and Related Work
===
The goal of this lab is for you to survey work related to your project, decide on the most relevant baselines, and start to implement them.

Ideally, the outcome of this lab would be: (1) the related work section of your project report is written and (2) baselines have been benchmarked.

Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>

1: Related Work
----
1. <a href="https://arxiv.org/pdf/1908.01073.pdf">U-Net Fixed-Point Quantization for Medical Image Segmentation</a> </br>

<b>1.1 Summary of Work:</br></b>
This paper proposed a fixed point quantization for UNet. Given a network with a full percision (float32), we can use the folloing fixed point quantization function to quantize the parameters (wieghts and activation) into n bits in the inference step: <br/>
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq11.png)</br>
where round function projects its input to the nearest integer, << and >> are shift left and right operators respectively. The clamp function is defined as the following:<br/>
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq12.png)</br>
To map any given number x to its fixed point value we can first split the number into its fractional(xf) and integer parts (xi), and then use the following equation to convert x to its fixed point representation using the specitied number of bits for the integer (ibits) and fractional (fbits) parts:<br/>
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq13.png)</br>
Since the clamp function is differentiable (under different ranges), we can train our network quantization using gradient descent. In the paper, it states that with only 4bits for weights and 6 bits for activations, UNet can achieve 8 fold reduction in memory while only losing less than 2.5% accuracy in the given dataset (3 datasets were used in the paper). </br>
<b>1.2. Inspiration for Project</br></b>
We believe the method this paper purposed can be directly replied to our UNET and strike a trade-off between accuracy and the required memory. Since the quantized model doesn't require floaing point computation, it will be more suitable on the edge devices given the constraint on CPUs and GPUs. </br>
 </br>
2. <a href="https://hal.inria.fr/hal-02277061/document">Low-power neural networks for semantic segmentation of satellite images</a> </br>
<b>2.1 Summary of Work:</br></b>
This paper proposed a compact version of UNet, which is called C-UNet. C-UNet drastically reduce the number of parameters by removing three convolutions stages (experimental validation result) and change several convolution layers to depthwise-seperable convolution layers. C-UNet has 1000 times less parameters than UNet, while its performance on 2 testing datasets (38-Cloud and CloudPeru2) is almost the same. Overall, this paper introduced lightweight UNet structure called C-UNet that uses less memory and FLOPs without compromising the accuracy on the given segmentaiton tasks.
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq11.png)
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq21.png)</br>
<b>2.2. Inspiration for Project</br></b>
This paper indicates that 1) we should emprically search the minimum number of stage reauired for the UNet without harming the performance drastically. 2) replace the convolution layers with depthwise-seperable layers to reduce the number of parameters. However, we doubt that these methods will not harm the performanceo on our tasks since we are using a way more complicated dataset compared to the binary image segmentation dataset used in the paper.</br>
 </br>
3. <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8999615">Low-Memory CNNs Enabling Real-Time
Ultrasound Segmentation Towards
Mobile Deployment</a> </br>
<b>3.1 Summary of Work:</br></b>
Similar to the previous paper "Low-power neural networks for semantic segmentation of satellite images" where reducing the stages and replacing convlution layers with depthwise-seperable layers are introduced in this work. It further applied Knowledge Distillation to restore the model performance. Specifically, in the paper they use a larger teached model (the thin, regular convolution network) to supervised the training of a lighter student network(the thin, seperable convolution network). The distillation method involves optimizing the student network with respect to the sum of 2 losses: a hard target and a distillation loss which encodeds information on the teacher's internal representation. For classification problems, the distillation loss often uses the teacher;s output as "soft targets", in an identical fashion to the manual labels. However, in the case of segmentation, the teacher's outputs offer negligible additional information over the manual lavels. As an alternative, this paper introduced a distillation loss using the teacher network's intermediate activations, incentivizing the student network to recreate the feature maps of the larger architecture. Overall, this paper purposed a 3 steps model optimization, reduce stages -> depthwise-seperable convlution -> knowledge distillation, for unet and shows that these optimization technique can make the final model runs 9x faster than the standard UNet on a CPU and requires 420x less space im memory without harning the task accuracy performance.   </br>
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq31.png)</br>
<b>3.2. Inspiration for Project</br></b>
This paper indicates clearly documents how to apply knowledge distillaion on UNet, and we believe it will be what we also want to apply for our UNet for the project. </br>
 </br>
4. <a href="https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Beheshti_Squeeze_U-Net_A_Memory_and_Energy_Efficient_Image_Segmentation_Network_CVPRW_2020_paper.pdf">Squeeze U-Net: A Memory and Energy Efficient Image Segmentation Network</a> </br>
<b>4.1 Summary of Work:</br></b>
This paper introduced "fire module", which is similar to the one in SqueezeNet to reduce the number of parameters of the original UNet by 16.84x. SqueezeU-Net inference speed is also 17% faster than original UNet without harming the performance of the tasks. Specifically, the fire modules are integrated into U-Net's contracting (downsample) and expansive(upsample) paths. The fire modules initial depthwise convolution reduce the number of channels and compensates this reduction by an inception stage with two parallel convolutions each having hald the number of output channels of the fire module's output channels. The two parallel convolutions help prevent feature loss and vanishing gradients which may be caused by reducing the number of channels. Overall, this work presents Squeeze Unet which has 12x fewer parameters and 3x fewer MACs. Squeeze U-Net use fire modules and tranpsoed fire modules in the contracting and expansive paths of U-Net to reduce model size and generate a memory and power efficient segmentation model. The
12× reduction in memory requirements also result in a significant reduction in energy requirement compared to UNet and the 3× reduction in MACs similarly should reduce
energy dissipation for computation.
</br>
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq41.png)</br>
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq42.png)</br>
![image](https://github.com/cwuu/11-767/blob/main/labs/lab5_image/eq43.png)</br>
<b>4.2. Inspiration for Project</br></b>
We should apply fire modules and transposed fire modules to our UNet for reducing the number of parameters and MACs. 

2: Baselines
----
1. What are the baselines that you will be running for your approach? Please be specific: data, splits, models and model variants, any other relevant information.
   </br><b>Ans: </b> 
   </br><b>Model: </b> UNet
   </br><b>Data: </b> Subset of the original test data - 25 RAW images (cropped to 512x512)
   </br><b>Splits: </b> N/A
   </br>(Due to the time limit, other baselines implementation are not completed)
   
2. These baselines should be able to run on your device within a reasonable amount of time. If you haven't yet tried to run them, please include a back-of-the-envelope calculation of why you think they will fit in memory. If the baselines will not fit in memory, return to (1) and adjust accordingly.
   </br><b>Ans: </b> On a 4GB Jetson Nano, the inference time for the 25 RAW images is 135 minutes and 24 seconds (~325s for an image)
   
3. How will you be evaluating your baselines?
   </br><b>Ans: </b> The focus is on the inference speed against the memory usage to see how compute and visual performance trade-off. For measuring the inference speed, we Will take more than 20 RAW images and test the run time. The process will include the complete pipeline (i.e. data preprocessing, inference and writing the result image out). Then the result will be the average processing time for an image.

4. Implement and run the baselines. Document any challenges you run into here, and how you solve them or plan to solve them.
   </br><b>Ans: </b>  The implementation is straight forward but there are some challenges when trying to run the model on the Jetson Nano. 
   </br>The first challenge is that the RAW image processing. There is no easy installation/wheel provided for rawpy (a python module for RAW image processing) on ARM64 platform. Therefore, we have to install the missing dependencies and to compile the package from the source code.
   </br>The second challenge is that the root cause of the program crash is not obvious. After analyzing the memory usage, it is found that the program consumed all the physical and swap memory (randomly). It is solved by increasing the swap memory of the Jetson nano. 
   
5. If you finish running and evaluating your baselines, did the results align with your hypotheses? Are there any changes or focusing that you can do for your project based on insights from these results?
    </br><b>Ans: </b> Yes, the results align with our hypotheses. We believe that UNet cannot fit into the physical memory completely and thus the inference speed will be impacted. The first thing we should focus on is to reduce the memory usage of the UNet. The second thing we may wish to try is to replace part of the model with traditional computer vision techniques.
    
3: Extra
----
More related work, more baselines, more implementation, more analysis. Work on your project.


FAQ
----
1. Our baseline is the SotA model from XXX 2021 which doesn't fit on device.  

Yikes! I think you meant to say -- "We used some very minimal off the shelf components from torchvision/huggingface/... to ensure our basic pipeline can run"

2. We're evaluating our baseline on accuracy only

I think you meant to say -- "We plan to plot accuracy against XXXX to see how compute and performance trade-off. Specifically, we can shrink our baseline by doing YYYY"
