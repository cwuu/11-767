Lab 9: Carbon footprint
===
The goal of this lab is for you estimate the carbon footprint of your class project.

Group name: 1 tsp of sugars and 3 eggs
---
Group members present in lab today: <b>Emily Wuu(cwuu), Raymond Lau(kwunfunl)</b>

1: Inference
----
1. Plug your device in to the Kill-a-watt and run inference using your model to get a measurement of the energy draw. What is its baseline energy draw, and how does that compare to running inference?

Answer:

| Configuration          | Energy Draw (Watt) |
|------------------------|--------------------|
| Baseline (UNet)        | 8.2                |
| Light (UNet)           | 6.3                |
| Lighter (UNet)         | 6.1                |

2. Multiply energy draw by inference time to get an estimate of energy required per inference (you can average over input size).

| Configuration          | Energy Draw (Watt) | Latency (s) | Energy per Inference (J) |
|------------------------|--------------------|-------------|--------------------------|
| Baseline (UNet)        | 8.2                | 94.74       | 776.868                  |
| Light (UNet)           | 6.3                | 89.06       | 561.078                  |
| Lighter (UNet)         | 6.1                | 76.94       | 469.334                  |

3. Multiply this by the carbon intensity of the location of your device. You can use [this resource](https://www.epa.gov/egrid/power-profiler#/).
- Device location: Shadyside, Pittsburgh PA
- EPA eGRID region: RFCW
- Carbon intensity: 1067.7 lbs CO2/MWh

| Configuration          | Energy per Inference (J) | CO2(lbs)      |
|------------------------|--------------------------|---------------|
| Baseline (UNet)        | 776.868                    | 0.0002304061 |
| Light (UNet)           | 561.078                    | 0.00016640638 |
| Lighter (UNet)         | 469.334                   | 0.00013919664|



4. Please include at least this estimate in your final project report.

2: Training
----
1. Did your project require training a model? If so, compute that estimate as well. If you used cloud resources, you can use [this tool](https://mlco2.github.io/impact/#compute) to help estimate. Otherwise, try to use the methods discussed last class for estmating carbon footprint due to training. Show your work and explain.

<br>Yes, for our project we need to re-train the model.
<br>We have the following assumptions/settings to estimate the power usage.:
<br>   1. 4000 epochs for each model variant
<br>   2. Ignore some overhead time (eg: powering up the machine, other background tasks etc) since the training consumes the most large portion of power.
<br>   3. Round to the closest hour
<br>The machine we used for training is one AWS instance, display card is Tesla V100-SXM2-16GB and the region is N.Virginia.

The following watts are estimated baesd on the tool provided (https://mlco2.github.io/impact/#compute)

| Model         | Time for training 1 epoch (s) | Training time for 4000 epochs(h)   | Carbon emitted (kg) |
| ------------- | ------------- | ------------- | ------------- |
| UNet-QAT      | 40.25  | 44.72 | 4.14 |
| UNet-Light    | 35.19  | 39.10 | 3.62 |
| UNet-Lighter  | 38.06  | 42.29 | 3.91 |

3: Extra
----
1. Everything else: Do you have logs you can use to estimate the amount of energy + carbon that went in to all project development? Other ways of estimating? Was your device plugged in and running for the entire semester?
2. Supply chain / lifecycle analysis: Can you estimate the additional footprint due to manufacturing hardware? Lifetime energy use of your project if it was deployed in the real world?
3. If you have a Pi or Jetson 4GB: Compare Kill-a-watt measurements to command line energy measurements, and discuuss.
