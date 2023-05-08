<!--- This is how a comment in Markdown looks like -->


## Distributed Training on Large Data

- ImageNet-1k : still gold standard in training large visual recognition models
- Serves as "Hello World" for large dataset training

\center{\includegraphics[width=0.8\textwidth]{../images/MNIST_CIFAR_ImageNet_Transition_Test.pdf}}

## Distributed Training on Large Data

- ImageNet-1k : still gold standard in training large visual recognition models
- ResNet-50 : baseline model network, test accuracies : $\approx 75\%$ top-1,  $\approx 94\%$ top-5 (Winner ILSVRC 2015)

\center{\includegraphics[width=0.8\textwidth]{../images/ImageNet_Accuracy_Detection_Top_Only_Test.pdf}}

^[\tiny Russakovsky et al, IJCV, 2015]

## Distributed Training on Large Data
- ResNets on ImageNet : efficient distributed training in data parallel mode possible
  - prerequisite is good scaling of throughput during training
  - image throughput during training ideally increasing as $\tau_K = K \cdot \tau_{ref}$ Images/sec
  - training with a large effective batch size $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, $K$ workers

\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster_Ims_SpeedUp_ResNet-152x4_mod.pdf}}

^[\tiny Cherti and Jitsev, arXiv:2106.00116, MedNeurIPS Workshop, 2021]

## Distributed Training on Large Data
- ResNets on ImageNet : efficient distributed training in data parallel mode
  - High test accuracy in the end of the training is the goal

\center{\includegraphics[width=0.6\textwidth]{../images/Yamazaki_RunTime_Table_ImageNet_ResNet-50_Accuracies.png}}

\center{ \hspace*{5cm} \includegraphics[width=0.4\textwidth]{../images/ResNet-50_ImageNet_Training_Time_Validation_Accuracy.png}}

\vspace*{-1cm}
^[\tiny Yamazoto et al, 2019; Ying, 2018]

## Distributed Training on Large Data

- Data parallel training: working with large effective batch sizes
- Reminder: Training with $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, $K$ workers
- Large effective batch sizes alter model optimization trajectory


\center{\includegraphics[width=0.9\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}

## Distributed Training on Large Data

- Data parallel training: working with large effective batch sizes
- Training with $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, $K$ workers
- Large effective batch sizes alter model optimization trajectory
  - may require hyperparameter re-tuning compared to a working smaller batch (single node) version

\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}

^[\tiny Goyal et al, 2017]

## Distributed Training on  Large Data
- ResNet-50 : efficient distributed training in data parallel mode
  - for very large batch sizes $\vert \mathfrak{B} \vert$: diminishing speed-up returns when training towards a given test accuracy

\center{\includegraphics[width=0.8\textwidth]{../images/Batch_Size_Critical_ImageNet_ResNet-50_Goal_Error.png}}

^[\tiny Shallue et al, JMLR, 2019]

## Distributed Training on  Large Data
- Critical large batch sizes $\vert \mathfrak{B}_{\text{crit}} \vert$: diminishing speed-up when crossing, given target test accuracy

\center{\includegraphics[width=\textwidth]{../images/Batch_Size_Critical_ResNet_Different_Datasets_Test.pdf}}

^[\tiny Shallue et al, JMLR, 2019]
<!-- $\vert \mathfrak{B}_{\text{crit}} \vert = K \cdot \vert B_{\text{ref}} \vert$ -->

## Distributed Training on  Large Data
- Critical large batch sizes $\vert \mathfrak{B}_{\text{crit}} \vert$: systematic evidence across datasets, tasks and architectures

\center{\includegraphics[width=\textwidth]{../images/Batch_Size_Critical_Transformer_Different_Datasets.pdf}}

^[\tiny Shallue et al, JMLR, 2019]

## Distributed Training on  Large Data
- Critical large batch sizes $\vert \mathfrak{B}_{\text{crit}} \vert$: large enough to do efficient distributed training
- Efficient Distributed Training with $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, for large $K$
  - providing almost linear training speed up, $t_{\mathfrak{B}} = \tfrac{1}{K} t_{B}$

\center{\includegraphics[width=\textwidth]{../images/Batch_Size_Critical_ResNet_Different_Datasets_Test.pdf}}

^[\tiny Shallue et al, JMLR, 2019]

## Distributed Training on Large Data

- Efficient Distributed Training with $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, for large $K$
- Providing almost linear training speed up, $t_{\mathfrak{B}} = \tfrac{1}{K} t_{B}$, **without loss of test accuracy**
- Important: reducing training **time to accuracy** - **time to solution**
  - strong scaling : reducing **time to accuracy**
  - reducing time per update step, per epoch, increasing samples throughput - alone not **sufficient** for speeding-up, reducing **time to accuracy**!
  - doing "bad" update steps during training would require doing a lot of them before reaching target loss/accuracy ...

\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}

^[\tiny Goyal et al, 2017]

<!-- -->

## Distributed Training on  Large Data
- Efficient Distributed Training with $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, for large $K$
- Hyperparameters tuning may allow for even larger batch sizes while still reducing time to accuracy

\center{\includegraphics[width=\textwidth]{../images/Batch_Size_Critical_Different_Optimizers.pdf}}

^[\tiny Shallue et al, JMLR, 2019]

## Distributed Training on ImageNet
- Efficient Distributed Training with $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$, for large $K$
- Combating accuracy loss when using larger batch sizes: hyperparameter tuning
- Reducing **time to accuracy** with **target accuracy** equal to a working smaller batch (single node) reference

\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}

## Distributed Training on ImageNet
- Combating accuracy loss when using larger batch sizes: hyperparameter tuning
- Learning rate rescaling with respect to $\vert \mathfrak{B} \vert$ and $\vert B_{\text{ref}} \vert$

\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}

# Combating accuracy loss on ImageNet
- Learning rate rescaling: motivation to match weight updates for different batch sizes $\vert \mathfrak{B} \vert$, $\vert B_{\text{ref}} \vert$, $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$


\center{\includegraphics[width=0.8\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}


# Combating accuracy loss on ImageNet
- Learning rate rescaling: motivation to match weight updates for different batch sizes, $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$
  - increase the weight update step size to accommodate for the fewer number of update steps when having a larger batch size

\center{

$K$ update steps of SGD with learning rate $\eta$ and $\vert B_{\text{ref}} \vert = n$:

$$ \mathbf{W}_{t+K} = \mathbf{W}_t - \underbrace{\eta \frac{1}{n}} \sum_{j<K} \sum_{X\in B_j} \nabla \underbrace{\mathcal{L}(X,\mathbf{W}_{t+j})} $$


Single update step with $\vert \mathfrak{B} \vert = Kn$, learning rate $\hat\eta$


$$ \mathbf{\hat W}_{t+1} = \mathbf{W}_t  -  \underbrace{\hat\eta \frac{1}{Kn}} \sum_{j<K} \sum_{X\in B_j} \nabla \underbrace{\mathcal{L}(X,\mathbf{W}_t)} $$
}

^[\tiny Goyal et al, 2017]

# Combating accuracy loss on ImageNet
- Learning rate: linear rescaling, $\hat\eta = K \eta$, for $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$

\vspace*{0.5cm}

\center{

To get $\mathbf{\hat W}_{t+1} \approx \mathbf{W}_{t+K}$,

\vspace*{0.5cm}

we assume $\nabla \mathcal{L}(X,\mathbf{W}_t) \approx \nabla \mathcal{L}(X,\mathbf{W}_{t+j})$ for $j<K$

and obtain $$ \hat\eta \tfrac{1}{kn} = \eta \tfrac{1}{n} \Leftrightarrow \hat\eta = \tfrac{kn}{n} \eta \Leftrightarrow \hat\eta = K \eta $$

}

^[\tiny Goyal et al, 2017]

## Combating accuracy loss on ImageNet
- Learning rate: linear rescaling, $\hat\eta = K \eta$, for $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$
  - used in combination with usual learning rate schedules
- $\nabla \mathcal{L}(X,\mathbf{W}_t) \approx \nabla \mathcal{L}(X,\mathbf{W}_{t+j})$ for $j<K$ does not hold in general
  - especially wrong for initial learning phase where gradients vary a lot from step to step
  - A possible remedy: initial warm-up phase   

\center{\includegraphics[width=0.45\textwidth]{../images/WarmUp_LearningRate_Schedules.pdf}}


## Combating accuracy loss on ImageNet
- Learning rate: linear rescaling, $\hat\eta = K \eta$, for $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$
  - used in combination with usual learning rate schedules
- $\nabla \mathcal{L}(X,\mathbf{W}_t) \approx \nabla \mathcal{L}(X,\mathbf{W}_{t+j})$ for $j<K$ is bad assumption for early learning
- Warm-up phase: start with $\eta$, increase towards scaled $\hat\eta = K \eta$ within few epochs

\center{\includegraphics[width=\textwidth]{../images/WarmUp_BatchSizes_ResNet_ImageNet_Goyal.pdf}}

^[\tiny Goyal et al, 2017]

## Combating accuracy loss on ImageNet
- Learning rate tuning: package of mechanisms
  - linear rescaling
  - Warm-up for initial epochs
  - Schedules

\center{\includegraphics[width=\textwidth]{../images/Learning_Rate_Schedule_Package_ImageNet_8k_Goyal}}

## Combating accuracy loss on ImageNet
- Learning rate tuning: package of mechanisms
- Often, still not enough for very large batch sizes  $\vert \mathfrak{B} \vert > 8192$
- Advanced Optimizers that provide further adaptive hyperparamer tuning during training

\center{\includegraphics[width=\textwidth]{../images/Learning_Rate_Schedule_Package_ImageNet_8k_Goyal.pdf}}


## Combating accuracy loss on ImageNet
- Advanced optimizers that provide further adaptive hyperparamer tuning during training: very large batch sizes  $\vert \mathfrak{B} \vert > 8192$
- LARS : Layer-wise Adaptive Rate Scaling, extension of SGD with momentum
  - tuning learning rates layerwise depending on gradient and weight amplitudes and norms
- LAMB : Layer Adaptive Moment Batch, extension of LARS (use AdamW as base)
  - tuning learning rate layerwise, also per weight parameter using gradient mean and variance

\center{\includegraphics[width=0.45\textwidth]{../images/LAMB_32k_ResNet_ImageNet.pdf}}

^[\tiny You et al, ICLR, 2020]

## Combating accuracy loss on ImageNet
- Learning rate rescaling, schedules and Warm up : works well for $\vert \mathfrak{B} \vert \leq 8192$)
- Advanced optimizers (LAMB) : works for $\vert \mathfrak{B} \vert \leq 80k$)
- Almost linear speed-up in training time without accuracy loss: reducing **time to accuracy**

\center{\includegraphics[width=\textwidth]{../images/Table_ImageNet_Training_BatchSizes.pdf}}

^[\tiny Osawa et al, 2020]

## Distributed training with large batches
- More advanced techniques may allow efficient distributed training beyond batch size issues  
- Local SGD: giving up consistency between model parameters across different workers after each update
- Post Local SGD: combining coupled global SGD and decoupled local SGD
- Natural SGD: attempt to use second derivatives and curvature information

\center{\includegraphics[width=0.8\textwidth]{../images/Forward_Backward_Data_Parallel_Workers_Batches.pdf}}


## Distributed training with large batches
::: block

### Summary
- Efficient data parallel training on large datasets like ImageNet-1k
- Measures to stabilize training with large batches necessary
- Learning rate scaling, schedules, warm-up phase, specialized optimizers
- Advanced methods required for very large $\vert \mathfrak{B} \vert \geq 32k$
- Aim: reduce **time to accuracy** without accuracy loss

:::

<!--
- Advanced optimizers required for very large $\vert \mathfrak{B} \vert \geq 32k$
- Further advanced methods for even larger batch sizes
-->

\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}
