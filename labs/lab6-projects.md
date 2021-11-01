Lab 5: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.

Group name:
---
Group members present in lab today:

1: Plan
----
1. What is your plan for today, and this week <br/>
1.1 Today: Install neccessary packages and dependencies <br/>
1.2 This week: Applied tricks we find in the literature survey to our UNet Models, including (fixed-point/ static/ dynamic) quantization, depthwise-seperable convolution, knowledge distillation, and fire modules to reduce the memory requirement for running the project. <br/>
1.3 This week: Investigate into bottleneck of inferencing. <br/>
1.4 This week: Define the quantitative and qualitative evaluation metrices for both image quality and inference speed <br/>

2. How will each group member contribute towards this plan?<br/>
Emily:
- Install neccessary packages and dependencies
- Depthwise-seperable convolution
- Fire modules 
- Define the quantitative and qualitative evaluation metrices for both inference speed

Raymond:
- Install neccessary packages and dependencies
- (fixed-point/ static/ dynamic) quantization
- Investigate into bottleneck of inferencing
- Define the quantitative and qualitative evaluation metrices for both image quality

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

2. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?
<br><br><b>Raymond</b>
<br>I tried to run the quantized model using the jit.load module of PyTorch. But somehow it pops up an erorr message:
<img width="1030" alt="Screen Shot 2021-11-01 at 3 45 42 PM" src="https://user-images.githubusercontent.com/90403016/139735622-df231703-d638-42bc-842b-4f510e3d97e6.png"> It is found that currently the self implemented lrelu function is not supported. I am going to use the official torch.nn.LeakyReLU and replace that.
<br>The second thing I tried was to use Numba (a JIT numpy acceleration module) to improve the execution efficiency of the slow prepcrossing step (which involved lots of numpy operations). It could not be compiled on the Jetson Nano platform. I was now trying to look for alternatives to replace that.

3. What were the contributions of each group member towards all of the above?
<br><b>Ans:</b>Each one of us explored different directions that are helpful for the project. Emily focused on improving the network structure while Raymond focused on quantization and preprocessing steps. We also found and defined different appropriate metrics for the evaluations.
  
3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.
2. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?
3. How will each group member contribute towards those steps? 
