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
1.2 This week: Bug fix the quantization scheme and fire modules. <br/>
1.3 This week: Based on the results, investigate any other bottleneck of inferencing. <br/>
1.4 This week: Inspect the output and see how the image quality change. Then brainstorm/propose any other new directions. <br/>
1.5 This week: Implmeneted depthwise-seperable layer into UNet and speed up the training. <br/>

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
<br>1. I have fixed the quantization error for lrelu. The model size has been reduced to 3.98x of the original size. Here are some results:

    | Model | Time for 25 (512x512) images |
    | --- | --- |
    | Original model | 325.00 |
    | Quantized model(qint8) | 51.11 |

    It is found that the inference speed has been improved by 6.36x. The reason is that the quantized model can now completely fit into the fast physical memory. For the original model, part of the model and data are stored in the slow virtual memory(i.e. SD card). It takes a long time to access the data and the model in the slow virtual memory. The memory becomes a bottleneck for the inference speed.  
<br>Attached some of the examples
![00012_00_train_250 0](https://user-images.githubusercontent.com/90403016/141134057-00d5cfaa-c739-419c-a90c-52c81e557e4a.jpg)
![00043_00_train_250 0](https://user-images.githubusercontent.com/90403016/141134166-31841c5e-4685-4e06-b4f3-83440a00fdd7.jpg)
<br>From the above outputs, the visual quality are satisfactory. I believe more works including the fire module can be further incoporate.

<br>2. Besides, the time for the pre-processing step, which involved lots of numpy concatenation, has been estimated. It is found that the time for one image is just around 0.25s, which is just around 0.5% for the whole processing time. Therefore, we will put focus on other area instead of compiling the Numba module.

<br><b>Emily</b>
<br>1. I integrated depthwise seperable layers into UNet. There are 2 types of UNet, one is light-UNet, which only have the first convlution layer in each stack become depthwise seperable layer and the other remain the same as the traditional convolution layer; the other is lighter-UNet, which both of the convolution layer in the stack become depthwise seperable layers. They are both still in the training stage. The following is the summary of the number of the parameters for each version of the UNets:<br>
| Model  | Num of Params |
| ------------- | ------------- |
| Original model  | 7760748  |
| Light-UNet  | 4285676 (1.81x smaller)  |
| Lighter-UNet  | 1513812 (5.13x smaller) |

<br>Here are some visulization results of Light-UNet(2000 iters/ 4000 iters):
 ![00012_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab7-img/light1.png)
![00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab7-img/light2.png)
<br>Here are some visulization results of Lighter-UNet(1100 iters/ 4000 iters):
 ![00012_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab7-img/lighter1.png)
![00043_00_train_250 0](https://github.com/cwuu/11-767/blob/main/labs/lab7-img/lighter2.png)

After the model is fully trained, we will integrated it with the quantization method to get the final number on how much we can improve on the inferecne speed.

2. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?
<br><br><b>Raymond</b>
<br>I hoped to implement the script for checking the Structural Similarity Index and the Peak signal-to-noise ratio. The formula of SSIM is complex and therefore I took sometime to understand the try to figure out the implementaiton. I believed that this would take a long time to implement that and therefore I am now seeking alternatives/implemented codes.

<br><br><b>Emily</b><br>


3. What were the contributions of each group member towards all of the above?
<br><b>Ans:</b>Each one of us explored different directions that are helpful for the project. Emily focused on improving the network structure while Raymond focused on quantization and preprocessing steps. We also found and defined different appropriate metrics for the evaluations.
  
3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.
<br><b>Ans:</b> 
From the preliminary results, we believe that we are making progress towards completing our final project. The reason is that the visual quality of output from the quantized model/with depthwise-seperable layers and fire module are satisfactory. Up to now, there is no unexpected outcome. Therefore, we can continue the directions and seek other techniques to optimize the inference speed/visual quality/metrics.

2. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?
 <br><b>Ans:</b> 
Implemented a unified script to find the quantized metrics. And to combine the depthwise-seperable layers and fire module with the quantization aware training. 
3. How will each group member contribute towards those steps? 
<br><b>Ans:</b> Raymond and Emily will equally take half of the training and evaluation for the experiments the ablation study. And will discuss and analyze the result together. 
