## Deep Learning \& Distributed Training

<!-- Deep Learning \& Distributed Training -->

- Training models that solve complex, real world tasks requires large model and data scale

\vspace{-0.2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/Compute_Budget_Training_Milestone_models_2022_mod.pdf}}

\vspace{-0.5cm}
<!--^[\tiny Amodei et al, 2019 (Source: https://openai.com/blog/ai-and-compute/)]
-->
^[\tiny Sevilla et al., 2023; OurWorldInData.org/artificial-intelligence; CC BY Source]


## Deep Learning \& Distributed Training
<!-- - Compute and memory demand of training increases rapidly
-->
- Compute demand increases exponentially, 3.4 months doubling time since 2012

\vspace{-0.1cm}
\center{\includegraphics[width=0.8\textwidth]{../images/Compute_Budget_Eras_FLOPS_2022_mod.pdf}}

^[\tiny Sevilla et al., arXiv:2202.05924, 2023]

## Deep Learning \& Distributed Training

- Networks: large models, many layers, large number of parameters (weights)
  - Vision: Convolutional, Transformer and Hybrid networks
  - hundreds of layers, hundred millions of parameters (currently up to 20B)

\center{\includegraphics[width=\textwidth]{../images/Vision_Networks_CoAtNet_2021_mod.pdf}}

^[\tiny Dai et al, NeurIPS, 2021]

## Deep Learning \& Distributed Training

- Networks: large models, many layers, large number of parameters (weights)
  - Language: Transformer networks
  - hundreds of layers, billions of parameters (GPT-3: 175 Billion)

\center{\includegraphics[width=0.65\textwidth]{../images/MegaTron_DeepSpeed_Microsoft_Turing_NLM_model-size-graph.jpg}}

\vspace*{-1cm}
^[\tiny Narayanan et al, 2021]

## Deep Learning \& Distributed Training

* GPT-3: 175 billion weights, $\approx$ 350 GB, does not fit on single GPU (A100: 40/80 GB)
* ResNets, ViT $<$ 1B weights, $\lesssim$ 40 GB, fit on single GPU
  - depending on chosen resolution of input $X$ and batch size $\vert B \vert$!

\center{\includegraphics[width=\textwidth]{../images/Network_Computation_Size_AlexNet.pdf}}



## Deep Learning \& Distributed Training

- GPT-3: 175 billion weights, $\approx$ 350 GB, does not fit on single GPU
- ResNets, ViT $<$ 1B weights, $\lesssim$ 40 GB, fit on single GPU
  - depending on chosen resolution of input $X$ and batch size $\vert B \vert$!


\center{\includegraphics[width=0.9\textwidth]{../images/Only_Possible_with_X_Parallelism.pdf}}
<!-- Forward_Backward_Loss_Weights_Updates.pdf -->

^[\tiny Awan et al., 2020]

## Distributed Training

- Use the computational power and memory capacity of multiple nodes of a large machine
- Requires taking care of internode communication

\center{\includegraphics[width=0.9\textwidth]{../images/MultiNode_Training_Accellerators.png}}

^[\tiny Ben-Nun \& Hoefler, 2018]

## Distributed Training

- Use the computational power and memory capacity of multiple nodes of a large machine
- Requires taking care of internode communication
- Requires high bandwidth interconnect between the compute nodes
  - HPC: InfiniBand (4$\times$ Mellanox 200 Gb/s cards on JUWELS Booster per node)
  - Not available on conventional clusters!

\center{\includegraphics[width=0.9\textwidth]{../images/MultiNode_Training_Communication_Collectives.png}}

^[\tiny Ben-Nun \& Hoefler, 2018]

<!-- \center{\includegraphics[width=0.9\textwidth]{../images/Only_Possible_with_X_Parallelism.pdf}}
-->

<!-- Multi node works, communication techniques -->

## Distributed Training

- Depending on whether full model fits on a single GPU, different schemes
  - data parallelism: split only data across GPUs, model cloned on each GPU
  - model parallelism: split within layers across GPUs
  - pipeline parallelism: split layer groups across GPUs


<!-- \vspace*{2cm} -->
\center{\includegraphics[width=\textwidth]{../images/Distributed_Training_Schemes_Workers.pdf}}

^[\tiny Laskin et al., 2020]

## Distributed Training

- Model does not fit on single GPU: no training without parallelization possible at all
  - AlexNet in 2012; GPT-3; CLIP ViT G/14, ...
- Model fits on single GPU: why distributed training?
  - multiple GPUs can drastically speed up training phase $\rightarrow$ **data parallelism**
    - e.g. ImageNet training: from days to hours or minutes

\center{\includegraphics[width=\textwidth]{../images/Distributed_Training_Schemes_Workers.pdf}}

^[\tiny Laskin et al., 2020]

## Distributed Training
- ImageNet distributed training: from days, to hours, to minutes

\center{\includegraphics[width=0.7\textwidth]{../images/Yamazaki_RunTime_Table_ImageNet_ResNet-50_Accuracies.png}}

\center{\includegraphics[width=0.45\textwidth]{../images/ResNet-50_ImageNet_Training_Time_Validation_Accuracy}}

\vspace*{-1cm}
^[\tiny Yamazoto et al., 2019; Ying, 2018]

## Deep Learning with Data Parallelism

* Data parallelism: simple approach for efficient distributed training
  - whole network model **has to** fit on one GPU: depends on batch size!
  - split whole dataset across multiple workers
  - speeds up model training – given good scaling and well-tuned learning
* Faster training, shorter experiment cycle – more opportunities to test new ideas and models

\center{\includegraphics[width=0.4\textwidth]{../images/ImageNet-1k_TPUs_Ims.png}}

^[\tiny Ying, 2018]

<!--  [show training speed ...] [scaling plots on JUWELS where ? end of Horovod session ...] -->

## Deep Learning with Data Parallelism

* Data parallelism: simple approach for efficient distributed training
  - whole network model **has to** fit on one GPU: depends on batch size!
  - split whole dataset across multiple workers
  - good scaling: necessary (but not sufficient!) for model training speed up
* Faster training, shorter experiment cycle – more opportunities to test new ideas and models

\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster_Ims_SpeedUp_ResNet-152x4_mod.pdf}}

^[\tiny Cherti and Jitsev, arXiv:2106.00116, MedNeurIPS Workshop, 2021]

## Deep Learning with Data Parallelism

  - Data parallelism: simple approach for efficient distributed model training
    - same model is cloned across $K$ workers
    - each model clone trains on its dedicated subset of total available data
    - synchronous (S-SGD) or asynchronous (A-SGD) optimization; central (parameter server) or decentralized (ring-AllReduce) communication for weight updates

\center{\includegraphics[width=0.8\textwidth]{../images/Data_Parallel_Scheme_Model_Workers_Processes_Data.pdf}}

## Deep Learning with Data Parallelism

- Data parallelism: simple approach for efficient distributed model training
  - can be understood as training a model using a larger mini-batch size $\vert \mathfrak{B} \vert$
    - $\mathfrak{B} = B_1 \cup \ldots \cup B_K$, $B_i \cap B_j = \emptyset,\ \forall  i,j \in K$ workers
    \vspace*{2mm}
    - $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, where $\vert B_{\text{ref}} \vert = n$ is original, reference batch size for a single worker

\center{\includegraphics[width=0.8\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}



## Reminder: Mini-Batch SGD

* Mini-batch SGD
  * perform an update step using loss gradient $\nabla_\mathbf{W} \mathcal{L}_B$ over a **mini-batch** of size $\vert B \vert = n \ll N$
$$ \nabla_\mathbf{W} \mathcal{L}_B = \nabla_\mathbf{W} \dfrac{1}{n} {\displaystyle \sum_{X_i \in B} \mathcal{L}_i} $$

  - update step: $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$

\center{\includegraphics[width=0.8\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}


<!--
FIXME next two slides repeat everything said on page 17 but in more detail.
      maybe cut page 17 in favor of these?
-->
## Deep Learning with Data Parallelism

* **Effective larger mini-batch** $\mathfrak{B}$ over $K$ workers
  * perform an update step using loss gradient $\nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}}$ over a larger **effective** mini-batch $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}}\vert,\ \vert B \vert = n \ll N$
$$ \nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}} = \nabla_\mathbf{W} \dfrac{1}{K} {\displaystyle \sum_{j = 1}^{K} } \dfrac{1}{n} {\displaystyle \sum_{X_i \in B_j} \mathcal{L}_i} = \nabla_\mathbf{W} \dfrac{1}{nK} {\displaystyle \sum_{X_i \in \mathfrak{B}} \mathcal{L}_i} $$
  - update step: $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}}$

\center{\includegraphics[width=0.8\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}


## Deep Learning with Data Parallelism

* Training a model using a larger mini-batch size $\vert \mathfrak{B} \vert$
  - $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, where $\vert B_{\text{ref}} \vert$ is original, reference batch size for a single worker
  - Update step: $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}}$
  - **Reminder:** Changes optimization trajectory and weight dynamics compared to smaller mini-batch training

\center{\includegraphics[width=0.8\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}


## Deep Learning with Data Parallelism

* Training a model using a larger mini-batch size $\vert \mathfrak{B} \vert$
  - $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, where $\vert B_{\text{ref}} \vert$ is original, reference batch size for a single worker
  - Update step: $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}}$
  - **Reminder:** Changes optimization trajectory and weight dynamics compared to smaller mini-batch training

\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}

^[\tiny Goyal et al., 2017]

## Deep Learning with Data Parallelism

* Data parallel distributed training requires:
  - **proper data feeding** for each worker
  - setting up workers, one per each GPU (model clones)
  - **sync of model clone parameters** (weights) across workers: update step – **communication load**
    - after each forward/backward pass on workers' mini-batches
    - $\nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}} = {\displaystyle \underbrace{\dfrac{1}{K}  \sum_{j = 1}^{K}}_{\text{across } K \text{ workers}} } \underbrace{\nabla_\mathbf{W}  \dfrac{1}{n} {\displaystyle \sum_{X_i \in B_j} \mathcal{L}_i}}_{\text{on worker } j}$

\vspace*{-1.5cm}
\center{\includegraphics[width=0.8\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}

## Deep Learning with Data Parallelism

* Data parallelism: proper data feeding for each worker
  - important not to let GPUs "starve" while training

\center{\includegraphics[width=0.9\textwidth]{../images/GPU_CPU_Data_Training_1.pdf}}

^[\tiny Figure: Mendonça, Sarmiento, ETH CSCS, 2020]

## Deep Learning with Data Parallelism

* Data parallelism: proper data feeding for each worker
  - important not to let GPUs "starve" while training

\center{\includegraphics[width=\textwidth]{../images/GPU_CPU_Data_Training_2.pdf}}

## Deep Learning with Data Parallelism

* Data parallelism: proper data feeding for each worker
  - data pipelines: handled either by
    - internal TensorFlow (see tutorial) or PyTorch routines
    - specialized libraries, e.g. NVIDIA DALI, WebDataset

```python
# Example for TensorFlow dataset API
import tensorflow as tf

[...]

# Instantiate a dataset object
dataset = tf.data.Dataset.from_tensor_slices(files)

[...]

# Apply input preprocessing when required
dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Randomize
dataset = dataset.shuffle(buffer_size)

# Create a batch and prepare next ones
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

[...]
```

## Deep Learning with Data Parallelism

* Data parallelism: proper data feeding for each worker
  - important not to let GPUs "starve" while training
  - data handling via **data pipeline** routines
  - use efficient **data containers**: HDF5, LMDB, TFRecords, WebDataset, ...

\center{\includegraphics[width=0.9\textwidth]{../images/GPU_CPU_Data_Training_1.pdf}}

<!-- FIXME almost the same as slide on page 22 -->
## Deep Learning with Data Parallelism

* Data parallel distributed training requires:
  - proper data feeding for each worker: *data pipelines, containers*
  - setting up workers, one per each GPU (model clones)
  - **sync of model clone weights** across workers: update step $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}}$
    - after each forward/backward pass on workers' mini-batches
    - $\nabla_\mathbf{W} \mathcal{L}_{\mathfrak{B}} = \dfrac{1}{K} {\displaystyle \sum_{j = 1}^{K} } \underbrace{\nabla_\mathbf{W}  \dfrac{1}{n} {\displaystyle \sum_{X_i \in B_j} \mathcal{L}_i}}_{\text{on worker } j}$

\vspace*{-1.5cm}
\center{\includegraphics[width=0.8\textwidth]{../images/Forward_Backward_Data_Parallel_Workers_Batches.pdf}}

## Deep Learning with Data Parallelism

* Data parallel distributed training requires:
  - proper data feeding for each worker: *data pipelines, containers*
  - **sync of model clone weights** across workers: handle communication between nodes
    - for large $K$ and large model size – high bandwidth required! Enter stage **InfiniBand** – HPC
    - efficient internode communication while training on GPUs! Enter stage **Horovod**, **PyTorch DDP**, ...

<!-- \vspace*{-1.5cm} -->
\center{\includegraphics[width=0.8\textwidth]{../images/Forward_Backward_Data_Parallel_Workers_Batches.pdf}}

## Deep Learning with Data Parallelism: Horovod
* Horovod: making data parallel distributed training easy
  - efficient worker communication during distributed training
    - synchronous, decentralized (**no** parameter servers, ring-AllReduce)
    - additional mechanisms like Tensor Fusion
  - works seamlessly with job managers (Slurm)
  - very easy code migration from a working single-node version

\center{\includegraphics[width=0.8\textwidth]{../images/Horovod_Logo_Training_Combined.pdf}}

\vspace{-0.5cm}
^[\tiny Sergeev, A., Del Balso, M., 2017]

## Deep Learning with Data Parallelism: Horovod
* Supports major libraries: TensorFlow, PyTorch, Apache MXNet
* Worker communication during distributed training
  - NCCL: highly optimized GPU-GPU communication collective routines
    - same as in MPI: \texttt{Allreduce}, \texttt{Allgather}, \texttt{Broadcast}
  - MPI: for CPU-CPU communication
  - Simple scheme: 1 worker – 1 MPI process
  - Process nomenclature as in MPI: \texttt{rank}, \texttt{world\_size}
  - for local GPU assignment: \texttt{local\_rank}

\center{\includegraphics[width=0.8\textwidth]{../images/Horovod_Logo_Training_Combined.pdf}}



## Deep Learning with Data Parallelism: Horovod
* Horovod: highly optimized library for data parallel distributed training
  - Name origin: "horovod" stands for east slavic (Ukraine, Russia, Bulgaria, Belarus, Poland, ...) circle dance form

\center{\includegraphics[width=0.4\textwidth]{../images/Horovod_Logo.pdf}}

## Distributed Training with Horovod

* Training model with a large effective mini-batch size:
  - $\mathfrak{B} = \bigcup_{i \le K} B_i$, $B_i \cap B_j = \emptyset,\ \forall  i,j \in K$; $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$
  - $B_{\text{ref}}$ is reference batch size for single worker

\center{\includegraphics[width=0.8\textwidth]{../images/Forward_Backward_Data_Parallel_Workers_Batches.pdf}}

## Distributed Training with Horovod
* Training loop: $K$ workers, one per each GPU
```
init: sync weights of all K workers
for e in epochs:
  shard data subsets D_j to workers j
  for B in batches:
    each worker j gets its own B_j (local compute)
    each worker j computes its own dL_j (local compute)
    Allreduce: compute dL_B, average gradients (communication)
    Update using dL_B for all K workers (local compute)
```

\center{\includegraphics[width=0.6\textwidth]{../images/Forward_Backward_Data_Parallel_Workers_Batches.pdf}}


## Distributed Training with Horovod
* User friendly code migration, simple wrapping of existing code
  - major libraries supported: TensorFlow, PyTorch, MXNet, ...

  ::: columns

  :::: {.column width=50%}

  ```python
  import tensorflow as tf
  import horovod.tensorflow.keras as hvd

  # Initialize Horovod
  hvd.init()

  [...]

  # Wrap optimizer in Horovod's DistributedOptimizer
  opt = hvd.DistributedOptimizer(opt)

  [...]
  ```

  ::::

  :::: {.column width=50%}

  ```python
  import torch
  import horovod.torch as hvd

  # Initialize Horovod
  hvd.init()

  [...]

  # Wrap optimizer in Horovod's DistributedOptimizer
  opt = hvd.DistributedOptimizer(opt)

  [...]
  ```

  ::::

  :::

## Distributed Training with Horovod
* Handled by dataset pipeline (Horovod independent): data sharding

\center{\includegraphics[width=\textwidth]{../images/Data_Parallel_Model_Workers_Processes_Data_Subsets.pdf}}


## Distributed Training with Horovod
* Handled by dataset pipeline (Horovod independent): data sharding

```python
# Example for TensorFlow dataset API
import tensorflow as tf
import horovod.tensorflow.keras as hvd

[...]

hvd.init()

# Instantiate a dataset object
dataset = tf.data.Dataset.from_tensor_slices(files)

[...]

# Get a disjoint data subset for the worker
dataset = dataset.shard(hvd.size(), hvd.rank())

[...]

# Randomize
dataset = dataset.shuffle(buffer_size)

# Create worker's mini-batch and prepare next ones
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

[...]
```

## Distributed Training with Horovod
* Create a Slurm job script for the code wrapped with Horovod
  - $K$ Horovod workers correspond to $K$ tasks in total, 1 MPI process each
  - $K = \text{nodes} \cdot \text{tasks-per-node}$ = $\text{nodes} \cdot \text{gpus-per-node}$

```bash
#!/bin/bash

#SBATCH --account=training2306
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu

srun python train_model.py
```

<!-- \center{\includegraphics[width=0.8\textwidth]{../images/Forward_Backward_Data_Parallel_Workers_Batches.pdf}}
-->
\center{\includegraphics[width=0.6\textwidth]{../images/Horovod_Logo_Training_Combined.pdf}}

## Distributed Training with Horovod

::: block

### Basics to parallelize your model

* Use Horovod to wrap existing model code
* Use data containers and pipelines to provide data to workers efficiently
* Create a Slurm job script to submit the wrapped code

:::

\center{\includegraphics[width=0.3\textwidth]{../images/Horovod_Logo.pdf}}


## Data Parallel Distributed Training
::: block

### Summary
- Opportunity to efficiently speed up training on large data
- Requires $K$ GPUs, the larger $K$, the better
- Training with a larger effective batch size $\vert \mathfrak{B} \vert = K \vert B_{\text{ref}} \vert$
- Data pipelines, high bandwidth network (InfiniBand) pave the way
- Horovod, PyTorch DDP, TensorFlow DistributedStrategies
- Additional measures to stabilize training – upcoming lectures

:::

\center{\includegraphics[width=0.65\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}
