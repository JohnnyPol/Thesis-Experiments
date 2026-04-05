# Experiment 1 Overview

Combined metrics dataset generated from all available `metrics.json` files.

## Included Runs

- exp1_1: Single Node Baseline | accuracy=89.74% | throughput=2.555 samples/s | avg_latency=0.383s
- exp1_2: Single Node Early Exit | accuracy=85.32% | throughput=3.740 samples/s | avg_latency=0.259s
- exp1_3: Homogeneous 2 Workers | accuracy=85.32% | throughput=3.106 samples/s | avg_latency=0.310s
- exp1_4: Homogeneous 3 Workers | accuracy=85.32% | throughput=2.908 samples/s | avg_latency=0.332s
- exp1_5: Heterogeneous Pi + Jetson | accuracy=85.32% | throughput=3.148 samples/s | avg_latency=0.306s
- exp1_6: Heterogeneous 2 Pis + Jetson | accuracy=85.32% | throughput=2.917 samples/s | avg_latency=0.331s

## Highlights

- Highest throughput: exp1_2 (Single Node Early Exit) at 3.740 samples/s
- Lowest total energy figure: exp1_5 (Heterogeneous Pi + Jetson) at 0.006448 kWh
