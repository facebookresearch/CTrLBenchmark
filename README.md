/!\ Doc in progress

# Continual Transfer Learning Benchmark
Benchmark aiming at helping research by investigating the behaviour of different models in the Lifelong Learning context.

## Creating a Stream:

The `TaskGenerator` class is at the center of the implementation of the CTrL Benchmark. 
It gives access to a high-level API allowing to seamlessly generate a wide variety of streams with a loose coupling between the different components such as the underlying dataset(s), the strategy to generate the tasks (split, incremental, mixture of datasets, ...) and the processing to apply to each task.   

The 3 main components of a `Task Generator` are:
- A pool of concepts to select from to generate the tasks. It can be a few classes, a full dataset or even a mixture of datasets.
- A pool of transformation that can be modified or combined to apply specific processing to the data for each task.
- A Strategy, describing how to combine the conecpts and trasnformation over time to generate an actual stream.

Each of these components can be created by hand or using our automatic `TaskGenerator` creation tool using yaml configuration files.

For examples simply executing
```python
import ctrl
task_gen = CTrl.get_stream('s_minus')
```
will return the corresponding task generator that be used either directy to generate tasks on the fly:
```python
t1 = task_gen.add_task()
t2 = task_gen.add_task()
t3 = task_gen.add_task()
...
```

or as an iterator:
```python
for t in task_gen():
    ...
```

### Available streams:
In the current version, only the streams of the CTrL benchmark are directly available, they can be obtained using the following names:
$S^+$: `"s_plus"`
$S^-$: `"s_minus"`
$S^{in}$: `"s_in"`
$S^{out}$: `"s_out"`
$S^{pl}$: `"s_pl"`
$S^{long}$: `"s_long"`


See the CONTRIBUTING file for how to help out.

# LICENSE
See the LICENSE file.
