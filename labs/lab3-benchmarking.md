Lab 2: Benchmarking
===
The goal of this lab is for you to benchmark and compare model inference efficiency on your devices. **You should benchmark 2*N* models or model variants, where *N* is the size of your group (so, two models per person.)** For now, if you don't have appropriate evaluation data in place that's fine; you can provide pretend data to the model for now and just evaluate efficiency.

Ideally, the models you benchmark will be related to and useful for your class project, but at the very least the efficiency metrics should be useful.

Include any code you write to perform this benchmarking in your Canvas submission (either as a link to a folder on github, in a shared drive, zip, etc).

Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>


1: Models
----
1. Which models and/or model variants will your group be benchmarking? Please be specific.<br/>
<b>Answer:</b> We use (1)Unet (prject model), (2)squeezenet, (3)mobilenet_v3_small, (4)resnet18
More specifically, here is how we initialize them:
```
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=32, pretrained=lab2_args['pretrained'])
    squeezenet = models.squeezenet1_0(pretrained=lab2_args['pretrained'])
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=lab2_args['pretrained'])
    resnet18 = models.resnet18(pretrained=lab2_args['pretrained'])
```
2. Why did you choose these models?
<br><b>Answer:</b> We choose unet as it is the model we will use for our course project. We use suqeezenet and mobilenet because they are canonical for deploying model on the edge, and we should refer to them as our goal. As for resnet, we pick it because it is commonly use in many scenario.
3. For each model, you will measure parameter count, inference latency, and energy use. For latency and energy, you will also be varying a parameter such as input size or batch size. What are your hypothesis for how the models will compare according to these metrics? Explain.
<br><b>Answer:</b> The hypothesis is that the inference latency and the energy use will increase when the number of parameter count/input size/batch size increase.

2: Parameter count
----
1. Compute the number of parameters in each model. Remember, in Torch you should be able to start with something like this:
   ```
   num_params = sum([np.prod(p.size()) for p in model.parameters()])
   ```
2. Does this number account for any parameter sharing that might be part of the model you're benchmarking? \\
<br><b>Answer:</b> Yes
3. Any difficulties you encountered here? Why or why not?
<br><b>Answer:</b> No as it is straight forward.
4. Result<br/>
 
Model           | UNet  |SqueezeNet  | MobileNet_v3_small | ResNet18 | 
--------------- |:----------:|:---------:|:--------:|:-------:|
*Parameter count* |  7763041      | 1248424      | 2542856      | 11689512     |  


3: Latency
----
1. Compute the inference latency of each model. You should do this by timing the forward pass only. For example, using `timeit`:
    ```
    from timeit import default_timer as timer

    start = timer()
    # ...
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282
    ```
    Best practice is to not include the first pass in timing, since it may include data loading, caching, etc.* and to report the mean and standard deviation of *k* repetitions. For the purposes of this lab, *k*=10 is reasonable. (If standard deviation is high, you may want to run more repetitions. If it is low, you might be able to get away with fewer repetitions.)
    
    For more information on `timeit` and measuring elapsed time in Python, you may want to refer to [this Stack Overflow post](https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python).
2. Repeat this, but try varying one of: batch size, input size, other. Plot the results (sorry this isn't a notebook):
   ```
   import matplotlib.pyplot as plt
   
   plot_fname = "plot.png"
   x = ... # e.g. batch sizes
   y = ... # mean timings
   
   plt.plot(x, y, 'o')
   plt.xlabel('e.g. batch size')
   plt.ylabel('efficiency metric')
   plt.savefig(plot_fname)
   # or plot.show() if you e.g. copy results to laptop
   ```
3. Any difficulties you encountered here? Why or why not?
   <br><b>Ans:</b> No, it is straight forward.
4. Result<br/>

(input_size, batch_size), meausred in seconds

Model           | UNet  |SqueezeNet  | MobileNet_v3_small | ResNet18 | 
--------------- |:----------:|:---------:|:--------:|:-------:|
*Latency*(224, 1) |  1.2671     | 0.29014      | 2.62070      | 0.387337    |  
*Latency*(224, 2) |  3.7161    | 0.71413     | 3.0517567     |  0.7201054    |  
*Latency*(224, 4) |  12.29219    | 2.2002     | 4.7435727     |  4.36507    |   
*Latency*(224, 8) |  24.13201    | 5.3923913     | 6.2334538     |  13.8161282    |  
*Latency*(224, 16) |  10.8370    | 2.488953450     | 1.449785689     |  2.90050734299    | 
*Latency*(224, 32) |  105.9710    | 9.99997    | 3.2001857200    |  5.739008712999    |
*Latency*(448, 1) |  6.07727805     |0.7682674      | 2.843582966      | 1.037830321   |  
*Latency*(448, 2) |  106.942194    | 3.129367     | 3.7858080299     |  5.31977    | 
*Latency*(448, 4) |  368.627    |  14.86     | 5.3856136     |  13.54240625    |   
*Latency*(448, 8) |  6976.361    | 11.68776    | 7.5792618     |  24.40143    |  
*Latency*(448, 16) |  2788.3400    | 22.65425    | 4.5588694     |  2.90050734299    | 

4: Energy use
----
1. Compute the energy use of each model. You can use the `powertop` tool on RPi and Jetson (must be run as root):
    ```
    sudo apt install powertop
    pip3 install powertop
    ```
    and/or the `jtop` tool on Jetson (see installation instructions [here](https://github.com/rbonghi/jetson_stats/)). Follow the same procedure as you used to compute latency, but this time compute energy: (avg) watts * time.

2. Any difficulties you encountered here? Why or why not?
<br><b>Answer: </b>Yes. As the jetson nano 2GB does not have the INA3221, we cannot use jtop/sysfs nodes to read the power consumption of the jetson. Therefore we cannot provide the energy consumption for this device.

3. Result
<br><b>Answer:</b> N.A.as mentioned above.

5: Discussion
----
1. Analyze the results. Do they support your hypotheses? Why or why not? 
<br><b>Answer:</b> (Can only comment on the latency due to the limitation of Jetson Nano 2GB) Yes. It is because the latency has a relationship with the number of parameters of the models. If a model has more parameters, it means that it requires more computations. Therefore, more time is needed to finish the computation. It then increases latency. <br>
The logic mentioned above also applies to input size and batch size.

5: Extra
----
A few options:
1. Compute FLOPs for each of your models. If you're using Transformer-based models, you might be able to use or modify the [`flops_counter.py`]() script in this directory. If you're using a CNN-based model, then you might be able to use or modify code in [this project](https://github.com/1adrianb/pytorch-estimate-flops) or similar. 
2. Evaluate on different hardware (for example, you might run the same benchmarking on your laptop.) Compare the results to benchmarking on your device(s).
3. Use real evaluation data and compare accuracy vs. efficiency. Describe your experimental setup in detail (e.g. did you control for input size? Batch size? Are you reporting average efficiency across all examples in the dev set?) Which model(s) appear to have the best trade-off? Do these results differ from benchmarking with synthetic data? Why or why not?

----
\* There are exceptions to this rule, where it may be important to include data loading in benchmarking, depending on the specific application and expected use cases. For the purposes of this lab, we want to isolate any data loading from the inference time due to model computation.
