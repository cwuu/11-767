Lab 7: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.


Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>


1: Plan
----
1. What is your plan for today, and this week <br/>
1.1 Today: Discuss the problems encountered and plan for the next sprint. <br/>
1.2 This week: Combine the QAT (quantization-aware training) with the depthwise seperable convolution. <br/>
1.3 This week: Based on the results, investigate any other bottleneck of inferencing. <br/>
1.4 This week: Inspect the output and see how the image quality change. Then brainstorm/propose any other new directions. <br/>

2. How will each group member contribute towards this plan?<br/>
Emily:
- Continue the developmen of fire modules.
- Inspect the results and propose any improvements.
- Estimate the memory usage and find any technique that can further applied.
- Investigate any other bottleneck of inferencing.

Raymond:
- Combine the depthwise seperable convolution with the QAT.
- Check the results and see if there is any problem when combining the two optimizations methods.
- Estimate the memory usage and find that if CUDA is applicable in this case.
- Investigate any other bottleneck of inferencing.

2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?
<br><b>Raymond</b>
<br>1. I have combined the quantization aware training with the depthwise seperable convolution. After around 3500 epochs of training, the results are comparable to the images with only the QAT. The difference in loss is around ~13%. It seems that we should be able to combine them together.

Here are some samples:
![00037_00_train_250 0](https://user-images.githubusercontent.com/90403016/142273346-d2327009-2403-4d63-96c6-87f2c71f805c.jpg)
![00084_00_train_300 0](https://user-images.githubusercontent.com/90403016/142273411-956631c2-bdb7-4cfc-8373-78391a2e8be2.jpg)



<br>2. However, there are some problems when trying to call the model and inference. The error message is shown below:
<img width="571" alt="Screen Shot 2021-11-17 at 1 45 54 PM" src="https://user-images.githubusercontent.com/90403016/142262968-90e07c58-9fc3-4917-8fb7-d00a15403d67.png">

 

<br><b>Emily</b>
<br>1. I integrated depthwise seperable layers into UNet. There are 2 types of UNet, one is light-UNet, which only have the first convlution layer in each stack become depthwise seperable layer and the other remain the same as the traditional convolution layer; the other is lighter-UNet, which both of the convolution layer in the stack become depthwise seperable layers. They are both still in the training stage. The following is the summary of the number of the parameters for each version of the UNets:<br>
| Model  | Num of Params |
| ------------- | ------------- |
| Original model  | 7760748  |
| Light-UNet  | 4285676 (1.81x smaller)  |
| Lighter-UNet  | 1513812 (5.13x smaller) |

<br>Here are some visulization results of Light-UNet(8000 iters/ 8000 iters):
 ![00012_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab8-img/light1.png)
![00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab8-img/light2.png)
[00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab8-img/light3.png)
<br>Here are some visulization results of Lighter-UNet(5500 iters/ 5500 iters):
 ![00012_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab8-img/lighter1.png)
![00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab8-img/lighter2.png)
![00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab8-img/lighter3.png)
<br>2. I wrote the evaluation script for measuring the quantitative results on the testing and training set. For the quantitative metrices, I picked SSIM, PSNR and MSE; for qualitative metrices, we plan to conduct a survey from real person to check which model generates the output that is best aligned with human's visual system.

2. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?
<br><br><b>Raymond</b>
<br>I hoped to run the inference on the Jetson but somehow an error pops up. The error message is not very meaningful and not be able to find any similar case on the internet. I will look into the quantization module of Pytorch and try to find a solution/fix the problem.
<br><br><b>Emily</b>
<br>The evaluation script now is suffered from memory usage, which required further optimization.


3. What were the contributions of each group member towards all of the above?
<br><b>Ans:</b>Each one of us explored different directions that are helpful for the project. Emily focused on improving the network structure while Raymond focused on quantization and preprocessing steps. We also found and defined different appropriate metrics for the evaluations.
  
3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.
<br><b>Ans:</b> Yes, we are making incremental progress. 


2. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?
 <br><b>Ans:</b> We will conduct the survey to check the human visual system for the quality of image reconstruction. Also, we will start the quantitative evaluation for the inference speed and energy usage.  
<br><b>Ans:</b> Raymond and Emily will equally take half of the training and evaluation for the experiments the ablation study. And will discuss and analyze the result together. 
