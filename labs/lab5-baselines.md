Lab 4: Baselines and Related Work
===
The goal of this lab is for you to survey work related to your project, decide on the most relevant baselines, and start to implement them.

Ideally, the outcome of this lab would be: (1) the related work section of your project report is written and (2) baselines have been benchmarked.

Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>

1: Related Work
----
1. Choose at least 2 pieces of related work per group member. For each piece of related work, write a paragraph that includes:
    - Summary of the main contributions of that work.
    - How your proposed project builds upon or otherwise relates to that work.

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
