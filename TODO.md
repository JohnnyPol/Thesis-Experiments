# TODO

## Experiments To Keep
- [*] 1.1 Single-node ResNet baseline.
- [*] 1.2 Single-node ResNet18EE and ResNet34EE.
- [*] 1.3 Three-worker ResNet18EE and ResNet34EE with this partitioning:
  - worker1: Layers 0 and 1
  - worker2: Layer 2
  - worker3: Layer 3
- [*] 2.0 Two models on three nodes, same idea as the current multi-model setup, but with changed early-exit layers:
  - ResNet18EE / ResNet18EE
  - ResNet34EE / ResNet34EE
- [*] 3.1 Keep the current Experiment 3 arrangement exactly as it is.
- [*] 3.2 Same Experiment 3 idea, but with a different partition arrangement.
- [*] 3.3 Same Experiment 3 idea, but with another different partition arrangement.


## Models
- [*] Add ResNet34 and its weights
- [*] Change the 3-way partitioning so that Layers 0 and 1 on the first node, Layer 2 on the second and the third layer on the third node. 
- [*] Add CIFAR100 dataset as well.

## Evaluation
- [ ] After running all experiments, ask Grigoris about the charts we need to focus on and the tables.
- [ ] Decide whether latency distributions should be analyzed per partition or per exit branch.
- [*] Remove network bytes from the metrics
- [ ] Organize better all the metrics


group all exp1.* 
average inference time per sample (latency per sample)
- table: experiment | accuracy | latency | throughput | model | exit0 exit1 exit2 exit3 - experiments 1.1 1.2 1.3 
- utilization (worker id - utilization) throughput, table:inference time (experiment, inference time), compare 1.2 1.3. 3 figures for 1.2 3 figures for
accuracy is deterministic
exp2:
- avg inference time
- node utilization for each, same as 1.3

exp3.1 3.2 3.3:
IT NEEDS A DIAGRAM
barplot utilization
- compare the experiments, find the best (subsection)
- avg inference time

## Writing / Thesis
- [*] Remove selective offloading from the thesis and anything to do with the cloud
- [*] Create a ResNet18EE and a ResNet34EE diagram to visualize how the input changes
- [*] Reference the libraries that you use, why you use them very concisely.
- [*] Shift the focus to the distribution of the multi-model setup.
- [*] Write down that we trained the models in a Jupyter Notebook
- [*] Write the specs of the pis, the packages that we use, and why we chose them, their purpose.


## Cleanup
- [ ] Implement Testing
- [ ] Update README files and the experiments descriptions on the repo
- [ ] Change the repo name into something more appropriate
- [ ] 
