Lab 3: Quantization
===
The goal of this lab is for you to benchmark and compare model size and inference efficiency **between quantized and original models** on your devices. You should benchmark the same models as you benchmarked last lab, so ideally **2*N* models or model variants, where *N* is the size of your group (so, two models per person.)** For now, if you don't have appropriate evaluation data in place that's fine; you can provide pretend data to the model for now and just evaluate efficiency.

Ideally, the models you benchmark will be the same as last class, but if you find that you're unable to run out-of-the-box quantization on your models, feel free to try quantizing other models, or a subset. Just be sure to explain what you tried, and why.

Include any code you write to perform this benchmarking in your Canvas submission (either as a link to a folder on github, in a shared drive, zip, etc).

Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>


1: Models
----
1. Which models and/or model variants will your group be studying in this lab? What is the original bit width of the models, and what precision will you be quantizing to? What parts of the model will be quantized (e.g. parameters, activations, ...)? Please be specific.
</br><b>Answer: </b>We use (1)Unet (prject model), (2)shufflenet, (3)mobilenet_v3_small, (4)resnet18
We used dynamic quantization for the models. Therefore weights and activations of those models were quantized.
<br>
<br>UNet - Original bit width: torch.float32/float32 
<br>Target Precision: int8
<br>ShuffleNet - Original bit width: torch.float32/float32
<br>Target Precision: int8
<br>Mobilenet_v3_small - Original bit width: torch.float32/float32
<br>Target Precision: int8
<br>Resnet18 - Original bit width: torch.float32/float32 
<br>Target Precision: int8

2. Why did you choose these models?
</br><b>Answer: </b>We choose unet as it is the model we will use for our course project. We use Shufflenet and mobilenet because they are canonical for deploying model on the edge, and we should refer to them as our goal. As for resnet, we pick it because it is commonly use in many scenario.

3. For each model, you will measure model size (in (mega,giga,...)bytes), and inference latency. You will also be varying a parameter such as input size or batch size. What are your hypotheses for how the quantized models will compare to non-quantized models according to these metrics? Do you think latency will track with model size? Explain.
</br><b>Answer: </b>The hypothesis is that the inference latency of the quantized models will be much lower than the non-quantized ones. Yes, we believe that latency will still track with the model size. Although the quantization reduced the complexity of the computations, the number of computation is still directly proportional to the model's size/paramters. Therefore, the latency will still increase when the model size increase.

2: Quantization in PyTorch
----
1. [Here is the official documentation for Torch quantization](https://pytorch.org/docs/stable/quantization.html) and an [official blog post](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) about the functionality. Today we'll be focusing on what the PyTorch documentation refers to as  **dynamic** quantization (experimenting with **static** quantization and **quantization-aware training (QAT)** is an option for extra credit if you wish). 
2. In **dynamic** PyTorch quantization, weights are converted to `int8`, and activations are converted as well before performing computations, so that those computations can be performed using faster `int8` operations. Accumulators are not quantized, and the scaling factors are computed on-the-fly (in **static** quantization, scaling factors are computed using a representative dataset, and remain quantized in accumulators). You can acitvate dynamic quantization to `int8` for a model in PyTorch as follows: 
   ```
   import torch.quantization
   quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```
3. Make sure you can use basic quantized models. Dynamic quantization using torchvision is quite straightforward. e.g. you should be able to run the basic [`classify_image.py`](https://github.com/strubell/11-767/blob/main/labs/lab3-quantization/classify_image.py) script included in this directory, which uses a quantized model (`mobilenet_v2`). If you are having trouble, make sure your version of torch has quantization enabled. This whl should work:
    ```
    wget herron.lti.cs.cmu.edu/~strubell/11-767/torch-1.8.0-cp36-cp36m-linux_aarch64.whl
    pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
    ```
4. Try changing the model to `mobilenet_v3_large` and set `quantize=False`. (Note that `quantize=True` may fail due to unsupported operations.) What happens?
5. Try to use this to quantize your models. If you're feeling lost and/or you're unable to get this to work on your model [here is a tutorial on using dynamic quantization on a fine-tuned BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html) and [here is one quantizing an LSTM language model](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html). 
</br><b>Answer: </b> For Mobilenet_v3_large without quantization (set quantize=False) will output the prediction as "bluetick", and for Mobilenet_v2 with quantization (set quantize = True) will output the prediction as "Great Dane". The difference here we believe is caused by drop of accuracy of the quantized model. By directly checking the input image, we feel the ground truth should be closer to "bluetick" instead of "grreat dane", which indicate that the performance of the quantized model drag. 

6. Any difficulties you encountered here? Why or why not?
</br><b>Answer: </b> To run the quantized model on jetson nano (ARM device), we need to set the torch.backends.quantized.engine = "qnnpack" ourselves.

3: Model size
----
1. Compute the size of each model. Given the path to your model on disk, you can compute its size at the command line, e.g.:
   ```
   du -h <path-to-serialized-model>
   ```
</br><b>Answer:</b>
</br>
| Model | Model Size - float32 (KB) | Model Size - int8 (KB) | Shrinkage (ratio)|
| --- | ----------- | ----------- |----------- |
| UNet | 1710.351 | 443.189 | 3.86 |
| ShuffleNet | 9286.327 | 6215.103| 1.49 |
| mobilenet_v3_small | 10303.615 | 5463.631 | 1.89 |
| resnet18 | 46836.375| 45301.151| 1.03|

2. Any difficulties you encountered here? Why or why not?
</br><b>Answer:</b>
</br> Since pytorch doesn't suppor dynamic quantization for UNet, we use static quantization for UNet.

4: Latency
----
1. Compute the inference latency of each model. You can use the same procedure here as you used in the last lab. Here's a reminder of what we did last time: 
   You should measure latency by timing the forward pass only. For example, using `timeit`:
    ```
    from timeit import default_timer as timer

    start = timer()
    # ...
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282
    ```
    Best practice is to not include the first pass in timing, since it may include data loading, caching, etc.* and to report the mean and standard deviation of *k* repetitions. For the purposes of this lab, *k*=10 is reasonable. (If standard deviation is high, you may want to run more repetitions. If it is low, you might be able to get away with fewer repetitions.)
    
    For more information on `timeit` and measuring elapsed time in Python, you may want to refer to [this Stack Overflow post](https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python).
2. Repeat this, varying one of: batch size, input size, other. Plot the results (sorry this isn't a notebook):
![Lab4_224x224](https://user-images.githubusercontent.com/90403016/137214827-4644c061-35e4-4210-861e-270ca16d5623.png)
![lab4_448x448](https://user-images.githubusercontent.com/90403016/137214838-1e908f36-b684-4127-88c5-734730ee65aa.png)

3. Any difficulties you encountered here? Why or why not?
</br><b>Answer:</b>Compare to the original model, the quanitzed ones run much faster. However, we observe that the inference speed does not always increase with the number of batch size but peak at batch size=8. This result is out of expectation, and is observed on both original and quantized models. Originally, we think the reason is because we use a for-loop to run the experiment with different batch size and input image dimension, and the system might fetch the data in advance to save time. Hence, we discard the for-loop but let every experiment has its own .py file to be excuted. However, it sill give us the same result. We believe this might caused by the CPU arhcitecture, which has the aligment conflict when the batchsize = 8, but this requires further investigation. 

5: Discussion
----
1. Analyze the results. Do they support your hypotheses? Why or why not? Did you notice any strange or unexpected behavior? What might be the underlying reasons for that behavior?
</br><b>Answer:</b> For models applied dynamic quantization, we didn't see the inference speed become faster. Only UNet, which we manually applied static quantization has clear speed up. We believe this is because pytorch only supports linear layer quantization but all of our networks is CNN-based, meaning that most of the layers haven't been quantized. This fact can somehow be reflected on the size of the model in the previous section as well. 

5: Extra
----
A few options:
1. Try to run static quantization, or quantization aware training (QAT). Benchmark and report your results. Here is a nice blog post on [using static quantization on Torchvision models](https://leimao.github.io/blog/PyTorch-Static-Quantization/) in PyTorch.
2. Compute FLOPs and/or energy use (if your device has the necessary power monitoring hardware) for your models. 
3. Evaluate on different hardware (for example, you might run the same benchmarking on your laptop.) Compare the results to benchmarking on your device(s).
4. Use real evaluation data and compare accuracy vs. efficiency. Describe your experimental setup in detail (e.g. did you control for input size? Batch size? Are you reporting average efficiency across all examples in the dev set?) Which model(s) appear to have the best trade-off? Do these results differ from benchmarking with synthetic data? Why or why not?

----
\* There are exceptions to this rule, where it may be important to include data loading in benchmarking, depending on the specific application and expected use cases. For the purposes of this lab, we want to isolate any data loading from the inference time due to model computation.
