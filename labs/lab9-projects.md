Lab 8: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.


Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>


1: Plan
----
1. What is your plan for today, and this week <br/>
1.1 Today: Discuss why the inference speed does not change much after depthwise seperable convolution and plan for the next sprint. <br/>
1.2 This week: Conduct more experiments on the depthwise seperable convolution and see if there is any change of the inference speed. <br/>
1.3 This week: Seek any alternative to further optimize the model while preserving the visual quality. <br/>
1.4 This week: Setup a google survey form/other medium to collect the qualitative results. <br/>

2. How will each group member contribute towards this plan?<br/>
Emily:
- Code the evaluation script to generate images collection for coduing google survey
- Code and run the calculation for quantitative evaluation metrics for the testing images  

Raymond:
- Solve the error/technical issue of QAT depthwise seperable convolution with the QAT.
- Try to use CUDA for inference.
- Check the results and see if there is any problem when combining the two optimizations methods.
- Investigate any other bottleneck of inferencing.

2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?
<br><b>Raymond</b>
<br>1. Looked into the problem of the following error message:
<img width="571" alt="Screen Shot 2021-11-17 at 1 45 54 PM" src="https://user-images.githubusercontent.com/90403016/142262968-90e07c58-9fc3-4917-8fb7-d00a15403d67.png">. 
<br>It is found that the root cause of this problem is the mismatch torch.vision package. Currently the model trained on the server using PyTorch 1.10 and torchvision 0.11. When the quantized module is saved using the JIT module, extra values (eg: hook, in this case) are stored. On the Jetson nano, the torchvision module is version 0.9. It cannot load the values stored and triggered the problem. After reinstallation of the torchvision module on the serverside, the network is able to be loaded in the Jetson Nano. 

<br>2. After solving the technical problem, it is fonud that inference time for the optimized model (QAT + Depthwise seperable convolution) does not change much.


| Model  | Time for 25 (512x512) images |
| ------------- | ------------- |
| Original model  | 325.00  |
| Quantized model(qint8)  | 51.11  |
| Quantized model(qint8) + depthwise seperable conv | 52.0 |

 <br> On the other hand, it is observed that the quality of the pictures do not change much even with the two optimisations.
 ![00049_00_train_100 0](https://user-images.githubusercontent.com/90403016/143138179-9f786e1c-110a-4987-acca-42d4e96c81e5.jpg)
![00083_00_train_300 0](https://user-images.githubusercontent.com/90403016/143138204-18e3caeb-84cc-41e3-97ee-dad8246f5ba3.jpg)



<br><b>Emily</b>
<br>1. Evaluation script to generate the qualitative results and quantitve metrics is completed. Demo is shown as the following and we will use the generated testing images to design the google survey.
![00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/lab9-img/test.png)

2. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?
<br><br><b>Raymond</b>
<br>I tried to use CUDA for the inference as the model size has been compressed. But it turns out that Jetson Nano does not support INT8 inference. It is due to the hardware constraint and hence not possible to solve it. One alternative is to use float16 quantization.
<br><br><b>Emily</b>
<br><br><b>Emily</b>
<br> The evaluation couldn't feed in single GPU's memory, which always make the program crash in the middle of evaluating our testing sets. I use cpu to run the evaluation program to get rid of this issue temporarily since our testing set is small. However, I plan to expedite the evaluatino using mutiprocessing as we have 64 cpu in total.
![00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/lab9-img/cuda_memory_issue.png)
3. What were the contributions of each group member towards all of the above?
<br><b>Ans:</b>Each one of us explored possible directions that can further optimize the original model. Raymond focused on increasing the level of optimizations and doing different experiments while Emily is preparingt the necessary scripts for benchmarking and survey. We also confirmed some qualitative and quantitative metrics for the evaluations.
  
3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.
<br><b>Ans:</b> Yes, we are making progress step-by-step.


2. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?
 <br><b>Ans:</b> We will try to further optimize the model using the lighter model (with more extreme) Then, we will try to generate more images (different settings) and conduct the survey to check the human visual system for the quality of image reconstruction. Besides, we will measure the energy usages for the 3 models (original, quantized + depthwise seperable convolution, quantized + extreme depthwise seperable convolution).

3. How will each group member contribute towards those steps? 
<br><b>Ans:</b> Raymond and Emily will equally take half of the training and evaluation for the experiments the ablation study. And will discuss and analyze the result together. 
