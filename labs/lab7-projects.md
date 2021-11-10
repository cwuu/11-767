Lab 5: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.


Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>


1: Plan
----
1. What is your plan for today, and this week <br/>
1.1 Today: Discuss the problems encountered and plan for the next sprint. <br/>
1.2 This week: Bug fix the quantization scheme and fire modules. <br/>
1.3 This week: Based on the results, investigate any other bottleneck of inferencing. <br/>
1.4 This week: Inspect the output and see how the image quality change. Then brainstorm/propose any other new directions. <br/>

2. How will each group member contribute towards this plan?<br/>
Emily:
- Continue the developmen of fire modules.
- Inspect the results and propose any improvements.
- Estimate the memory usage and find any technique that can further applied.
- Investigate any other bottleneck of inferencing.

Raymond:
- Trouble shoot the problem encountered in quantization-aware training
- Find a method to compile Numba on ARM system.
- Inspect the results and propose any improvements.
- Estimate the memory usage and find any technique that can further applied.
- Investigate any other bottleneck of inferencing.

2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?
<br><b>Raymond</b>
<br> - Compiled a necessary Rawpy package(no wheel for ARM64 system) on the Jetson Nano. <br>
<br> - Explored static, dynamic and QAT quantization. Implemented a training script to apply Quantization-aware training for the model. Trained the model using the QAT schemes for 2000 epochs. <br>
<br> - Identified two bottlenecks for the inference: 
<br>     1. There is a slow preprocessing for step which involved manipulatio of the numpy arrays. 
<br>     2. The size of the model is large such that part of it is located in the slow virtual memory.<br>
<br> - For the quantitative evaluation of image quality, I found SSIM and PSNR are good indicators/metrics. As for the qualitative evaluation, one appropriate way is to use human to judge the quality of the output.<br>

<br><b>Emily</b>
<br> - Implement light-Unet and lighter-Unet with depthwise-seperable convlutional layer. The difference between light-Unet and lighter-Unet is light-Unet only has the second conv layer in each stack become depthwise-seperable convlutional layer, and the other one is a regular 3x3 convolutional layer; lighter-Unet has both con lyaer becone depthwise-seperable convolutional layer.   <br>
<br> - Implement the training and evalution script for light-Unet and lighter-Unet. Training has started.  <br>
<br> - For the quantitative evaluation of inference speed, I will use (1) inference latency (2) energy usage. Additionally, we will also looks into the space reauirement for our model such as number of model parameters.<br>

2. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?
<br><br><b>Raymond</b>
<br>I tried to run the quantized model using the jit.load module of PyTorch. But somehow it pops up an erorr message:
<img width="1030" alt="Screen Shot 2021-11-01 at 3 45 42 PM" src="https://user-images.githubusercontent.com/90403016/139735622-df231703-d638-42bc-842b-4f510e3d97e6.png"> It is found that currently the self implemented lrelu function is not supported. I am going to use the official torch.nn.LeakyReLU and replace that.
<br>The second thing I tried was to use Numba (a JIT numpy acceleration module) to improve the execution efficiency of the slow prepcrossing step (which involved lots of numpy operations). It could not be compiled on the Jetson Nano platform. I was now trying to look for alternatives to replace that.
<br><br><b>Emily</b><br>
Currently, the training speed of light-Unet and lighter-Unet is extremely slow. Since we have to do lots of evaluation after the model is converged, I hope to speed up the trianing as much as possible by chaning the training script to ddp (```nn.parallel.DistributedDataParallel```).

3. What were the contributions of each group member towards all of the above?
<br><b>Ans:</b>Each one of us explored different directions that are helpful for the project. Emily focused on improving the network structure while Raymond focused on quantization and preprocessing steps. We also found and defined different appropriate metrics for the evaluations.
  
3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.
<br><b>Ans:</b> We believe that we have a resonable project schedule since both of us completed almost half of our project workload. However, we may spend more time on integrating each of our optimization tricks as we believe there might be dependency among them.
2. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?
 <br><b>Ans:</b> We will add investigating into the dependency of each optimization tricks (ablation study) as our next steps. 
4. How will each group member contribute towards those steps? 
<br><b>Ans:</b> Raymond and Emily will equally take half of the training and evaluation for the experiments the ablation study. And will discuss and analyze the result together. 
