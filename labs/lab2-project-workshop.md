Lab 2: Project Workshopping / Peer Feedback
===
The goal of this lab is for you to give and receive peer feedback on project outlines before writing up your proposal. 

- **You can find your team's reviewing assignments in the first sheet [here](https://docs.google.com/spreadsheets/d/1_pw_lYkFutMjuL1_j6RdxNyQlj7LvF_f5eEKr1Qm-w0/edit?usp=sharing).**
- **The second sheet contains links to all teams' outlines to review.**
- **Once you have reviewed an outline, please enter a link to your review (e.g. a Google Doc, public github repo, etc.) replacing your team name in the corresponding other team's row in the third sheet.**


Here's a reminder of what your completed project proposal should include:
- Motivation
- Hypotheses (key ideas)
- How you will test those hypotheses: datasets, ablations, and other experiments or analyses.
- Related work and baselines
- I/O: What are the inputs and output modalities? What existing tools will you use to convert device inputs (that are not a core part of your project) to a format readable by the model, and vice versa?
- Hardware, including any peripherals required, and reasoning for why that hardware is needed for this project. (This is where you will request additional hardware and/or peripherals for your project!)
- Will you need to perform training off-device? If so, do you need cloud compute credits (GCP or AWS), and how much?
- Potential challenges, and how you might adjust the project to adapt to those challenges.
- Potential extensions to the project.
- Potential ethical implications of the project.
- Timeline and milestones. (Tip: align with Thursday in-class labs!)

Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>

1: Review 1
----
Name of team being reviewed: <b>The Hagrids</b>

1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's background is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
- Our team has members having experience on deploying models on computational resources constrained hardware.
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation?
- Hypothesis and testing
  - It is nice to have the reference paper and propose some directions (i.e. pruning , distillation) :-)
  - It will be better to have some preliminary testing to see how the original model works/behaves on the Jetson nano.
  - Would be nice to have more elaboration on the speaker identification system and how it can benefit the actual use case.

- I/O
  - Concise and clear about the input and output. Besides, the bandwidth of the system seems able to handle all of those.
  - I really like how the team clearly describes how they will preprocess the input data and present the output data. 

- Potential Challenges
  - Agree that hardware will be a potential problem. Just in most of the times, there will be some developed libraries/API to handle those.
  - As stated in the proposal, it is possible that some libraries are not running well on the ARM based system. There will be no wheel provided and it will not be able to be installed using the “pip install” command. 
  - For this problem, we would like to recommend you to find the source code of the library and then recompile it on the system. It is likely that it will solve the problem.

- Potential extension
  - Quite confused with this one. Is it going to be run on the same board with 2 microphones and using a single microphone? Or are the users speaking at the same time/separate time frame? If the speakers are speaking in separate time frames, we cannot see the advantage of having this extension.

3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
- Yes, we think the team keeps the scope pretty well.  
4. Are there any potential ethical concerns that arise from the proposed project?
- No, since the team will use the existing dataset.
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful? 
- N/A


2: Review 2
----
Name of team being reviewed: <b>MasterOfScience (MSC)</b>
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's backgorund is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.) 
- We have some background in Computer Vision. While this project also contains microphone input and multi-modality, this may be a bit complicated.
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation? 
- The motivation is clear. It will be better to have more explanations/descriptions to test the hypothesis.
E.g.:
For the Hypothesis 2 - “Provide end-to-end real-time inferencing at the edge”, we are not so sure if this is going to work on the edge device. As real-time usually means 30fps for video/camera. Unless the model is light, it is not easy to achieve that standard.
 
- Recommend to test the inference speed of the target network first. Then base on that to estimate the target speed/fps.


3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
- Not so sure about which model is going to be implemented. Therefore it is difficult to judge if the project is scoped properly. It will be better to provide/describe the basic model that is going to be implemented.

4. Are there any potential ethical concerns that arise from the proposed project? 
- Not so much. As the team mentioned that the inference is purely based on the edge device, the data will not go online.
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
- There might be some problems for the I/Os. As stated in the outline, there will be 2 cameras and 2 microphones. These devices may bring heavy workload to the bandwidth of the system. In that case, it will further impact the inference speed. Besides, reading the data of those inputs will consume the CPU resources.

- Recommend to test the CPU and ram usage for the single camera and microphone first.


3: Review 3
----
Name of team being reviewed: <b> Macrosoft </b>
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's background is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
- Our team has members with a CV background (face recognition, action recognition…). 
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation?
- General:
  - Since there is a great portion of the course addressing model optimization, we think it will be great to see how you might want to utilize these techniques. 

- Hypothesis & Testing:
  - Need more elaboration on what’s the constraint of the dataset you want to create. This can be from multiple dimensions, something related to the classroom setting can be square-feet and layout of people involved; something related to the sensor setting can be the image resolution you captured and the length of conversation. All of these should help reduce the changing factors as much as possible. This will help get the workable solution in the very first place.
  - For localizing the speaker, might also try vision-based method (eg. triangulation…) to see which one gives you the best accuracy.

- Potential Challenge:
  - It seems like there are many components involved in this project (real-time face detection, action recognition, and audio embedding). I would suggest trying to break down the project into smaller parts for testing purposes. For example, is it guaranteed that we can run real-time face detection on a Jetson nano? How much memory consumption for this module? Similarity for action recognition and audio embedding. Try to make sure each module can fit on the embedding devices first, and then work for the integration part.
  - It is also worth thinking about if you can share the backbone network for different tasks and just change the header for the downstream tasks. This can be helpful for reducing the computational resources required. 

3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
- We think there are too many components involved in the project. The team needs to collect the dataset themselves and work on the fusion of CV and NLP, which is quite a lot of work. It will be important for the team to try to think about how to break down the entire project into small segments and test them individually, or even try to think about a simplified version of the proposal. 
4. Are there any potential ethical concerns that arise from the proposed project?
- No, since the team will collect the dataset themselves and speech and image data captured only involved in team members. 
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful? 
- N/A


4: Receiving feedback
----
Read the feedback that the other groups gave on your proposal, and discuss as a group how you will integrate that into your proposal. List 3-5 useful pieces of feedback from your reviews that you will integrate into your proposal:
1. <b>"A confusing point is that the problems raised in the test hypothesis section talk about memory transfer speed, however the problem raised in the hypothesis section is about fitting the model and image into memory".</b> <br>
This is a good point as the relationship betwen the memory transfer speed and the memory location is not explained well in the proposal. Since the network is large (we have tried to ruin an inference on the Jetson), the physical memory(RAM) is not able to store all of it. Part of the data is moved to a low speed SD card. In that case, the low memory transfer speed is a bottleneck to the inference. We will make this clear when drafting the proposal. <br>
2. <b>"Just a few things to consider: are you sure that U-Net model + 1 mb image takes up 4 gb  of space? U-Net is only 250 mb".</b><br> Base on the theoretical values, these are correct numbers. One different is that when PyTorch model is being ran, there are soem memory overheads for the framework. We tested that the total memory consumption can go up to 3GB. We should add a screen capture to demonstrate the actual memory usage in the proposal.<br>
4. <b>"In the hypothesis section, they mentioned that “Phones' cameras do not perform well under low light conditions.mainstream cell phones nowadays share strong computation power, not to mention the iPad equipped with M1 chip".</b><br> The message we want to deliver is that we would like to ultilize the powerful computational resources on those phone and the deep learning model to enhance low light pictures. We will state that more clear in the proposal<br>
5. <b>"While the team identifies optimization as a challenge, instead of looking forward to an iterative approach which might potentially lead to unpredictable performance trends"</b><br> This is indeed a problem as it is likely to have unpredictable performance due to the optimization. Therefore, we would be more conservative and will use an iterative approach to do that. The aim is to try to find a balance point between acceptable visual performance and the model size. This will be written into the proposal.


You may also use this time to get additional feedback from instructors and/or start working on your proposal.


