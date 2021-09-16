Lab 1: Project Brainstorming
===
The goal of this lab is for you to work as a group to brainstorm project ideas. At the end of this exercise you should have an outline for your project proposal that you can share with others to get feedback. We will share these outlines for peer feedback in the next lab.

Group name:
---
Group members present in lab today:

1: Ideas
----
Write down 3-5 project ideas your group considered (a few sentences each). Depending on your group and process, the ideas may be very different, or they may be variations on one central idea.
 1. Try to increase the inference speed and reduce the size of the ViT mmodel (Reference paper: AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE)
 2. Deploy a camera running on mobile devices that can process extreme low-light photography by reducing the size of U-Net (paper: Learning to See in the Dark)
 3. Compress VAE
 4. Deploy a high accuracy vision-based tracker that can run in real-time on embedding devices by compressing CNN-based backbone.
 5. ...

2: Narrowing
----
Choose two of the above ideas. For each one:
1. How would this project leverage the expertise of each member of your group?
2. In completing this project, what new things would you learn about: (a) hardware (b) efficiency in machine learning (c) I/O modalities (d) other?
3. What potential road blocks or challenges do you foresee in this project? How might you adjust the project scope in case these aspects present unsurmountable challenges?
4. How could you potentially extend the scope of this project if you had e.g. one more month?

<b>Idea 2:</b>
   1. Emily has some experiences in UNet and Raymond has developed applications on embedded systems.
   2. Completing this project, we will learn about the ARM embedded systems, unified memory and the architecture of GPU. Besides, we will have some hands on experiences in the parallel computing.
   3. For a compressed network, the outputs may not have similar visual results as the original model. In that case, we may reduce the compression rate and tune the network until the final result is acceptable.
   4. If we achieve success in the Jetson platform, we may port the model to the Android/IPhone. This will enhance the camera functions of the phones.


3: Outline
----
Choose one of the ideas you've considered, and outline a project proposal for that idea. This outline will be shared with other groups next class (Tuesday) to get feedback.

Your outline should include:
- Motivation
- Hypotheses (key ideas)
- How you will test those hypotheses: datasets, baselines, ablations, and other experiments or analyses.
- I/O: What are the inputs and output modalities? What existing tools will you use to convert device inputs (that are not a core part of your project) to a format readable by the model, and vice versa?
- Hardware, including any peripherals required, and reasoning for why that hardware was chosen for this project. (This is where you will request additional hardware and/or peripherals for your project!)
- Potential challenges, and how you might adjust the project to adapt to those challenges.
- Potential extensions to the project.

