Lab 1: Project Brainstorming
===
The goal of this lab is for you to work as a group to brainstorm project ideas. At the end of this exercise you should have an outline for your project proposal that you can share with others to get feedback. We will share these outlines for peer feedback in the next lab.

Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>

1: Ideas
----
 1. Try to increase the inference speed and reduce the size of the ViT mmodel (Reference paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)
 2. Deploy a camera running on mobile devices that can process extreme low-light photography by reducing the size of U-Net (paper: Learning to See in the Dark)
 3. Compress VAE for facial 3D recontruction
 4. Deploy a high accuracy vision-based tracker that can run in real-time on embedding devices by compressing CNN-based backbone.

2: Narrowing
----
Choose two of the above ideas. For each one:
1. How would this project leverage the expertise of each member of your group?
2. In completing this project, what new things would you learn about: (a) hardware (b) efficiency in machine learning (c) I/O modalities (d) other?
3. What potential road blocks or challenges do you foresee in this project? How might you adjust the project scope in case these aspects present unsurmountable challenges?
4. How could you potentially extend the scope of this project if you had e.g. one more month?


<b>Idea 4:</b>
   <br>
   Ans 1: Emily has worked on some objects tracking projects before and Raymond has worked on images classficaiton tasks.
   <br>
   Ans 2: We will have more insights in the backbone and how does the backbone affect the performance of tracking. We may also have to think of multi-threading for I/O (i.e. Camera).
   <br>
   Ans 3: Compressing the mainstream backbones may not work. We may have to find another suitable backbone/work on other parts of the end-to-end model.
   <br>
   Ans 4: Further compress the backbone and try to find the limit of that.

<b>Idea 2:</b>
   <br>
   Ans 1: Emily has some experiences in UNet and Raymond has developed applications on embedded systems.
   <br>
   Ans 2: Completing this project, we will 1.learn about the ARM embedded systems, unified memory and the architecture of GPU 2. have hands-on experinece on seperable-depthwise convolution, model quantization, and optimization techniques from parallel computing (numerical and optimization algorithm on GPU).
   <br>
   Ans 3: For a compressed network, the outputs may not have as good visual results as the original model. In that case, we may reduce the compression rate and tune the network until the final result is satisfactory.
   <br>
   Ans 4: If the result is good on Jetson platform, we may port the model to the Android/IPhone. This will enhance the camera functions of the phones.
   <br>

3: Outline
----
<b>Deploy a camera running on mobile devices that can process extreme low-light photography by reducing the size of U-Net (paper: Learning to See in the Dark)</b>

- Motivation: 
 <br>Ans: Due to hardware CMOS limitaiton, Phones' cameras do not perform well under low light conditions. Hence, we would like to leverage deep learning model to improve the visual results on the final post-processed images. 
  
- Hypotheses (key ideas)
<br>Ans: We will use a pre-trained model as our baseline inference model, and use the dataset provided (containing raw images) as the input for the baseline inference model. Since the input images is specifically for tackling extra low-light condition, and the designed pre-trained model is also customized for this task, the most important hypothese for our project is that we can <b>run our baseline inference model with at least one input raw image on jetson nano (2GB) (regardless of the inference speed in the initial stage).</b> 
<br>Our hypothese is verifiable, and we can verify it by checking the memory usage during model inferencing with batch size of 1. With this correct hypothese, we will apply different model quantization skills, including but not limited to change the conventional cnn to depthwise seperable convolution, model quantization and model distallaion. 

- How you will test those hypotheses: datasets, baselines, ablations, and other experiments or analyses.
<br> Ans: To verify our hypothesis, we quicky check the memory usage on our current jetson nano (2GB), and find out that for loading the model with one raw images (1024x1024) on cpu, it takes 4.5 gb memory in total, where 1.9 gb is runing on jetson nano's 2gb main memory and 2.5 gb is running on sd card memory. Since the transmit speed on swaping memory between sd card and jetson nano's main memory is only ~ 20Mib/s, this greatly decreases the performance of inference speed. The memory usage of idle stage is 800MB for jetson nano's main memory, and 600MB for SD card. If we want to mount our model on GPU, this will actually double our memory usage. Considering these facts, to have our inference model squeeze on jetson nano main memory instead of borrowing from swapping between memory from sd card, we request to change our jetson nano from 2gb to 4gb for conducting a comprehensive experiment on model optimization, <b>we request to change to jetson nano(4gb).</b> 
- I/O: What are the inputs and output modalities? What existing tools will you use to convert device inputs (that are not a core part of your project) to a format readable by the model, and vice versa?
<br>Ans: Input:  Image (.ARW), Output: Image (.JPEG), Tools for converting device inputs: openCV and PyTorch/Tensorflow

- Hardware, including any peripherals required, and reasoning for why that hardware was chosen for this project. (This is where you will request additional hardware and/or peripherals for your project!)
<br>Ans: Normal bayer pattern sensor (Camera). The deep learning model aims at optimizing the low light performance of a specific architecture of CMOS. Hence, it is required to have the specified camera. 
- Potential challenges, and how you might adjust the project to adapt to those challenges.
<br>Ans: The original model is using a large and complex model to generate the satisfying results. It is likely that if we significantly compress the model, the final results are not well. Hence it is better to use an iterative approach to achieve that and find the limit.
- Potential extensions to the project.
<br>Ans: Port the compressed model to IPhones/Android phones.

