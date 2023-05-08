<!--- # Large Scale Deep Learning Motivation and Basics DL Recap
-->

<!--- Author : Jenia Jitsev, Dec 2020.
Contributors:
Reviewers: Jan Ebert
-->

<!--- UPDATE: Jenia Jitsev, Feb 2022.
Contributors:
Reviewers:
-->

<!-- Comments:

Why make everything a list?
Take a look at FIXMEs for graphics.
-->

<!-- **** PART 1 : Motivation Large Scale Deep Learning  -->

<!--- This is how a comment in Markdown looks like -->

<!--- # Scalable Deep Learning: Motivation -->


## Deep Learning: Current State of The Art in ML
- Breakthroughs in broad range of challenging domain problems
  - computer vision, language understanding, complex control, protein structure prediction, ...

\center{\includegraphics[width=\textwidth]{../images/Overview_Applications.pdf}}

^[\tiny Karras et al., 2019; Esteva et al., 2017; Jumper et al., 2020]

<!--^[\tiny Jumper et al., 2020] -->

## Deep Learning: Current State of The Art in ML
- Consistently outperforming all other ML methods on large dataset benchmarks
  - ImageNet-1k (1.4 M samples, 224x224, 0.4 TB), FFHQ (70k samples, 1024x1024, 2.5 TB), ...
  - Recent development: very large scale multi-modal (language-vision) datasets: LAION-400M/5B (400M/5B image-text pairs)

<!-- \vspace*{1cm} -->

\center{\includegraphics[width=0.7\textwidth]{../images/ImageNet_Outperforming.pdf}}


## Deep Learning: Current State of The Art in ML
- Consistently outperforming all other ML methods on large dataset benchmarks
  - convolutional and transformer networks predominant

<!-- \center{\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{../images/ImageNet_Outperforming_2.pdf}}
-->

<!-- \center{\includegraphics[scale=0.5]{../images/ImageNet_Outperforming_2.pdf} \hspace*{4cm}}  -->

\center{\includegraphics[width=0.9\textwidth]{../images/ImageNet_Outperforming_2.pdf}}

## Deep Learning: Current State of The Art in ML
- Self-supervised multi-modal (language-vision) learning: no explicit labels required
  - openAI CLIP: very strong zero- and few-shot transfer across various targets
  - requires very large data for pre-training (eg. publicly available LAION-400M/5B)
  - self-supervised learning from weakly aligned image-text pairs: public Internet as scalable data source

\center{\includegraphics[width=\textwidth]{../images/CLIP_Zero_Shot_Few_Shot_Self-Supervised_mod.pdf}}

^[\tiny Radford et al., ICML, 2021]

## Deep Learning: Current State of The Art in ML
- Consistently outperforming all other ML methods
  - AlphaFold 2: Transformer networks (predominant in natural language processing)
  - Big Fantastic Database (BFD): 2.4B protein sequences, 0.27 TB

\center{\includegraphics[width=\textwidth]{../images/CASP_Outperforming.png}}

^[\tiny Jumper et al., Nature, 2021]

## Deep Learning is Supercomputing
<!--- \framesubtitle{Deep Learning is Supercomputing} -->

- Training of models requires accelerators
  - GPUs (currently NVIDIA dominant), TPUs (Google)
  - GPUs: generic deep learning hardware
    - parallelizing matrix/tensor operations via vectorization

\center{\includegraphics[width=\textwidth]{../images/GPU_TPU.pdf}}

## Deep Learning is Supercomputing
<!--- \framesubtitle{Deep Learning is Supercomputing} -->

- Training of models requires accelerators
  - spezialized hardware, eg. in-memory computing chips
  - Graphcore IPU: Colossus MK2
  - Cerebras: Wafer Scale Engine 2 (850k cores!)

\center{\includegraphics[width=\textwidth]{../images/Graphcore_Cerebras.pdf}}

## Deep Learning is Supercomputing
<!--- \framesubtitle{Deep Learning is Supercomputing} -->

- Most breakthroughs require heavy compute power, using many accelerators simultaneously
  - GPT-3: natural language generation, language understanding
  - CLIP, DALL-E 2, Stable Diffusion: image understanding and image generation  
  - AlphaFold 2: protein structure prediction
  - AlphaZero, MuZero: learning control in highly dimensional state-action spaces

\center{\includegraphics[width=\textwidth]{../images/GPT-3_AlphaFold2_HighRes.png}}

## Deep Learning is Supercomputing
<!--- \framesubtitle{Deep Learning is Supercomputing} -->

- Many recent breakthroughs require heavy compute power, using many accelerators simultaneously
  - GPT-3: natural language generation, language understanding
  - CLIP, DALL-E: image understanding and image generation  
  - AlphaFold 2: protein structure prediction
  - AlphaZero, MuZero: learning control in highly dimensional state-action spaces

\center{\includegraphics[width=0.8\textwidth]{../images/JUWELS_Booster.pdf}}

## Deep Learning is Supercomputing

::: columns

:::: {.column width=40%}

- GPT-3: natural language generation, language understanding
  - 1 model: 175 billion weight parameters
  - compare:
    - AlexNet, winner ILSRVC 2012 – 60M
    - ResNet-50, winner ILSRVC 2015 – 25M
    - CLIP ViT L/14, multi-modal learning, 2021 - 600M

::::

:::: {.column width=60%}

<!-- \vspace*{8cm} -->

\vspace*{2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster.pdf}}

::::

:::

## Deep Learning is Supercomputing

::: columns

:::: {.column width=40%}

- GPT-3: natural language generation, language understanding
  - 1 model: $\approx$ 350 GB memory for training
  - splitting over 22 V100 GPUs required for one model to run

::::

:::: {.column width=60%}

<!-- \vspace*{8cm} -->

\vspace*{2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster.pdf}}

::::

:::

## Deep Learning is Supercomputing

::: columns

:::: {.column width=40%}

- GPT-3: natural language generation, language understanding
  - 1 model full training:\
    $\approx$ $3.14\cdot 10^{23}$ FLOPS
  - $\approx$ 355 years for V100;\
    $\approx$ 90 years for A100

::::

:::: {.column width=60%}

<!-- \vspace*{8cm} -->

\vspace*{2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster.pdf}}

::::

:::

## Deep Learning is Supercomputing

::: columns

:::: {.column width=40%}

- GPT-3: natural language generation, language understanding
  - 1 model full training:\
    $\approx$ $3.14\cdot 10^{23}$ FLOPS
  - $\approx$ 16 days if scaled well with $2\,000$ A100 GPUs on JUWELS Booster

::::

:::: {.column width=60%}

<!-- \vspace*{8cm} -->

\vspace*{2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster.pdf}}

::::

:::

## Deep Learning is Supercomputing

::: columns

:::: {.column width=40%}

- CLIP: self-supervised image-text learning, strong zero-shot transfer
  - openCLIP ViT g/14, 1480 A100 GPUs
  - 8 days on 34B samples from LAION-2B

  → for training a single language-vision model
::::

:::: {.column width=60%}

<!-- \vspace*{8cm} -->

\vspace*{2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster.pdf}}

::::

:::

## Deep Learning is Supercomputing

::: columns

:::: {.column width=40%}

- AlphaFold 2: protein structure prediction
  - 128 TPUs
  - few weeks

  → for training a single model
::::

:::: {.column width=60%}

<!-- \vspace*{8cm} -->

\vspace*{2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster.pdf}}

::::

:::

<!-- \center{\includegraphics[width=0.6\textwidth]{../images/JUWELS_Booster.pdf}} -->

## Deep Learning is Supercomputing
<!--- \framesubtitle{Deep Learning is Supercomputing} -->

- Foundation models: transferable models pre-trained on large generic data
  - transfer across domains specific smaller datasets and tasks
  - strong, efficient transfer: requires large pre-trained models

\center{\includegraphics[width=0.8\textwidth]{../images/Pretrain_Transfer_Foundation_Model_mod.pdf}}


<!-- \center{\includegraphics[width=0.6\textwidth]{../images/JUWELS_Booster.pdf}} -->

## Deep Learning is Supercomputing
<!--- \framesubtitle{Deep Learning is Supercomputing} -->

- Investigating strong transferable models: pre-training on a large dataset, requires a large machine
  - a rather compact ResNet-50 on ImageNet-1k: still **$\boldsymbol{\approx}$ 20 hours** for full training (single V100)
  - can be done in **a few minutes** if using thousands of GPU
  - High Performance Computing (HPC): many strong compute nodes, fast interconnect (InfiniBand)

\center{\includegraphics[width=0.8\textwidth]{../images/JUWELS_Booster.pdf}}

<!-- \center{\includegraphics[width=0.6\textwidth]{../images/JUWELS_Booster.pdf}} -->

<!-- **** PART 2 : Deep Learning Basics Recap -->
## Deep Learning: Basics Recap

::: block
### Machine Learning

Optimizing loss (objective) of a (complex) model $f$ using (a lot of) data $\mathcal{D}$

:::

- a (complex) model: function (or distribution) family $f(X;\boldsymbol{\theta})$ ($p(X;\boldsymbol{\theta})$)
    - parameters $\boldsymbol{\theta}$ (often $\mathbf{W}$ is used) are to adapt ("fit") given the data samples $X \in \mathcal{\hat{D}}$
- optimization:
    - defining a loss $\mathcal{L}(f(X;\boldsymbol{\theta}),\mathcal{\hat{D}})$
    - loss $\mathcal{L}$: measure of quality ("fit") of $f(X;\boldsymbol{\theta})$ in terms of a task solution on $\mathcal{\hat{D}}$
    - seeking to minimize $\mathcal{L}(f,\mathcal{\hat{D}})$ with respect to all possible $\mathcal{\hat{D}} \sim P(\mathcal{D})$!

\center{\includegraphics[width=\textwidth]{../images/Basic_ML_Optimization.pdf}}

<!--
- General Principle of Expected Risk minimization
- $f^* = \arg \min_{f} \mathbf{R}(f) = \mathbb{E}_{\mathcal{D} \sim P(\mathcal{D})}[\mathcal{L}(f, \mathcal{D})] = \int \mathcal{L}(f, \mathcal{D})\,dP(\mathcal{D})$  

-->

<!-- # Deep Learning as Optimization: Basics Recap -->

## Deep Learning: Basics Recap

::: block
### Machine Learning

Optimizing loss (objective) of a (complex) model $f$ using (a lot of) data $\mathcal{D}$

:::

- model $f(X;\mathbf{W})$: unknown recipe for "solutions" to a "problem" posed by $\mathcal{L}$
- optimization: looking for a "good" model $f^{*}$ by minimizing $\mathcal{L}(f,\mathcal{\hat{D}})$ for all possible $\mathcal{\hat{D}} \sim P(\mathcal{D})$!
    - general principle of expected risk minimization:

    \center{$f^* = \arg \min_{f} \mathbf{R}(f) = \mathbb{E}_{\mathcal{\hat{D}} \sim P(\mathcal{D})}[\mathcal{L}(f, \mathcal{\hat{D}})] = {\displaystyle \int \mathcal{L}(f, \mathcal{\hat{D}})\,P(\mathcal{\hat{D}})\,d\mathcal{\hat{D}}}$}

\center{\includegraphics[width=\textwidth]{../images/Basic_ML_Optimization_Data.pdf}}
<!-- FIXME in graphic on the left, the D describing the whole blob should be P(D) -->

<!-- , $\mathbf{W}$ to adapt -->
## Machine Learning: Generalization

::: block
### Important
Estimate loss $\mathcal{L}(f(X;\mathbf{W}),\mathcal{D})$ on data $\mathcal{D}$ yet unseen!

:::

- estimating "true" $\mathcal{L}$, expected risk: **generalization** error
  - Challenge: limited data – "true" $\mathcal{L}$, true loss landscape, true data distribution $P(\mathcal{D})$ unknown


\center{$f^* = \arg \min_{f} \mathbf{R}(f) = \mathbb{E}_{\mathcal{\hat{D}} \sim P(\mathcal{D})}[\mathcal{L}(f, \mathcal{\hat{D}})] = {\displaystyle \int \mathcal{L}(f, \mathcal{\hat{D}})\,P(\mathcal{\hat{D}})\,d\mathcal{\hat{D}}}$}

\center{\includegraphics[width=\textwidth]{../images/Basic_ML_Optimization_Data.pdf}}

## Machine Learning: Generalization

::: block
### Important

* Estimate $\mathcal{L}(f(X;\mathbf{W}),\mathcal{D}^{unseen})$ : aiming for good **generalization** capability

:::

- General approach: split $\mathcal{\hat{D}}$ into disjoint $\mathcal{D}^{tr}$ and $\mathcal{D}^{ts}$, $\mathcal{D}^{tr} \cap \mathcal{D}^{ts} = \emptyset$ !
  - learn, “train” on $\mathcal{D}^{tr}$: **training data set**
    - \alert{Do not show} $\mathcal{D}^{ts}$ - **test data set** - during training!
  - estimate $\mathcal{L}$, **generalization error** on $\mathcal{D}^{ts}$ after training


\center{\includegraphics[width=\textwidth]{../images/Data_Split_Optimization.pdf}}

## Machine Learning: Generalization

::: block
### Dataset split

* Estimate $\mathcal{L}(f(X;\mathbf{W}),\mathcal{D}^{unseen})$
  - Requires strict separation of data used for **any** parameter adaptation (training) from data for testing  

:::

- Often, model has further **hyperparameters** $\boldsymbol{\Theta}$ (eg. learning rate, ...)
  - adapt $\mathbf{W}$, then $\boldsymbol{\Theta}$: requires distinct sets
  - training $\mathcal{D}^{tr}$, **validation** set $\mathcal{D}^{val}$
- Estimate $\mathcal{L}$, **generalization error** on separate $\mathcal{D}^{ts}$
  - **never** use $\mathcal{D}^{ts}$ for parameter tuning followed by model selection, **only** for reporting metrics (loss, accuracy, ...)


## Machine Learning: Generalization

::: block
### Remember

*  $\mathcal{L}^{tr}$ measured on $\mathcal{D}^{tr}$ can be highly misleading for the quality of the trained model

* Minimizing $\mathcal{L}^{tr}$ is **not** the ultimate goal

:::

- **Overfitting**: very low $\mathcal{L}^{tr}$ on $\mathcal{D}^{tr}$, much higher $\mathcal{L}^{ts}$ on $\mathcal{D}^{ts}$
- **Generalization gap**
- Ultimate aim: minimize **generalization error**
  - “bad” outcomes on unseen data matter

\center{\includegraphics[width=0.8\textwidth]{../images/Generalization_Loss_Model_Complexity_Double_Descent.png}}


<!--

- Remember: loss/performance measured on $\mathcal{D}^{tr}$ can be highly misleading for the quality of the trained model
  - Overfitting: very low $\mathcal{L}^{tr}$ on $\mathcal{D}^{tr}$, much higher $\mathcal{L}^{ts}$ on $\mathcal{D}^{ts}$
  - Generalization Gap
  - Ultimate aim: minimize generalization error
    - “bad” outcomes on unseen data matter
    - minimizing $\mathcal{L}^{tr}$ **not** the ultimate target

-->

<!--

Learning as optimization problem
- Defining loss / cost function L sets up optimization problem
- The most important part is to estimate loss L on data yet unseen!

- Aim is to minimize generalization error : “bad” outcomes on unseen data

Problem: how to estimate generalization error? We see only performance on available data previously shown to A (training error)

- General approach: split D into Dtrain and Dtest
  - Learn, “train” on Dtrain (training data set). Hide Dtest from A
  - Estimate generalization error, “test”, on Dtest (test data set)

Performance measured on Dtrain can be highly misleading
- Overfitting: very high model fit on training data, but severe performance degradation on test data Dtest

Cross-validation: a popular technique to estimate  generalization error
- create number of different splits in training and test set
- average over different error estimates
- expensive for large problems

-->

## Machine Learning as Optimization Problem

- Optimization procedure: minimizing $\mathcal{L}(X;\mathbf{W})$ by adapting $\mathbf{W}$ from $\mathcal{D}^{tr}$

  - corresponds to **searching for a yet unknown, good model** $f(X;\mathbf{W})$ – data-driven tuning of $f(X;\mathbf{W})$

- Different optimization methods depending on problem setting, model class, loss

  - evolutionary search (ES), genetic algorithms
  - expectation-maximization (EM)
  - coordinate descent
  - gradient-based (first, second order gradient descent): requires **differentiable** $\mathcal{L}$!
  - etc...

\center{\includegraphics[width=0.95\textwidth]{../images/Optimization_Various.png}}

## Deep Learning as Optimization Problem

::: columns

:::: {.column width=40%}


* Deep Learning: using a generic, **differentiable** flexible model family $f(X;\mathbf{W})$
  - “neural” networks with multiple adaptive "layers", large number of those
  - based on stacked, adaptive, repetitive **differentiable** operations
  - non-linear operations executed by simple, generic units – a great number of those
  - Large number of weights $\mathbf{W}$ in the layers: adapted from incoming data

* Important: **differentiable** $f$ allows for **differentiable**  $\mathcal{L}$
  - using **simple first order gradient descent** methods turned out to be very successful

::::

:::: {.column width=60%}


\center{\includegraphics[width=\textwidth]{../images/DeepLearning_Basic_Network_UnitSpan.pdf}}
<!-- FIXME "eg," -> "e.g." -->

::::

:::

## Deep Learning: Gradient Descent

::: columns

:::: {.column width=40%}

* First order gradient descent (GD)
  - first derivatives (the gradient) of loss $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ are sufficient

  $$ \nabla_\mathbf{W} \mathcal{L} = \dfrac{\partial \mathcal{L}}{\partial \mathbf{W}} = \left( \begin{array}{c}
  \partial \mathcal{L} / \partial w_1   \\
  \vdots \\
  \partial \mathcal{L} / \partial w_m
  \end{array} \right) $$

  - use gradients to update the weights: $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}$
    - $\boldsymbol{\eta}$: update step size, **learning rate**

  - moving in direction of decreasing $\mathcal{L}$: $\mathcal{L}(f(\mathbf{W}_{t+1})) < \mathcal{L}(f(\mathbf{W}_t))$ (if $\Delta \mathbf{W}$ small)

::::

:::: {.column width=60%}

\center{\includegraphics[width=0.6\textwidth]{../images/Loss_Surface_Gradient_Descent_Walk.pdf}}

::::

:::


## Deep Learning: Gradient Descent

* Training: Loop until $\mathcal{L}^{tr}$, $\mathcal{L}^{ts}$, generalization gap are sufficiently small
  - **forward** pass, **backward** pass (**backpropagation** via **autodiff**, computing $\nabla_\mathbf{W} \mathcal{L}$ automatically), applying weight updates, ...


* Thanks to **automatic differentiation** (reverse mode), computation of $\nabla_\mathbf{W} \mathcal{L}$ for *any* differentiable $\mathcal{L}(f)$ via dedicated libraries (TensorFlow, PyTorch, ...)

\center{\includegraphics[width=0.8\textwidth]{../images/Forward_Backward_Loss_Weights_Updates.pdf}}



## Deep Learning: Gradient Descent

* Convolutional and Transformer networks: dominant in many domains
  - Convolutional: vision (ResNets, EfficientNets, RegNets, ...)
  - Transformers: language (GPT, BERT), vision (ViT, DeIT), language-vision (CLIP, DALL-E)
  - most compute intensive: tensor-tensor operations
  <!--
  - strong inductive bias (prior about the problem) via local spatial convolutional kernels
  - large reduction in weight size compared to fully connected networks
  - ResNet, DenseNet, ...
  -->
* Tensor operations, simple non-linearities
  - highly optimized **vectorized** implementations for GPU
    - cuBLAS, cuDNN
  - both forward and backward path: extremely efficient execution on GPUs – fast training

\center{\includegraphics[width=\textwidth]{../images/CONV_GPU.pdf}}


<!-- * Transformer networks: more generic tensor operations, optimized versions for cuDNN -->

## Gradient Descent as Optimization Procedure

::: columns

:::: {.column width=40%}

* Loss over a training set $\mathcal{D}^{tr}$ of size $N$ as sum of losses for $N$ single examples $X^i \in \mathcal{D}^{tr}$
  - remember: decomposition into sum due to data maximum likelihood estimation – i.i.d. assumption!

$$ \mathcal{L} = \underbrace{\dfrac{1}{N} {\displaystyle \sum_{i=1}^{N} \mathcal{L}_i}}_{\text{i.i.d.!}} = \dfrac{1}{N} {\displaystyle \sum_{i=1}^{N} \mathcal{L}(X^i, f(X^i;\mathbf{W}))} $$


- Differentiation is a linear operator. Loss gradient over all $X^i$:
$$ \nabla_\mathbf{W} \mathcal{L} = \nabla_\mathbf{W} \dfrac{1}{N} {\displaystyle \sum_{i=1}^{N} \mathcal{L}_i} = \dfrac{1}{N} {\displaystyle \sum_{i=1}^{N} \nabla_\mathbf{W} \mathcal{L}_i} $$

::::

:::: {.column width=60%}

\center{\includegraphics[width=0.6\textwidth]{../images/Loss_Surface_Gradient_Descent_Walk.pdf}}

::::

:::

<!-- * weight updates: direction opposed to gradient, always decrease loss, moving to a minimum

$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t - \eta \nabla_\mathbf{w} \mathcal{L}$
-->

<!--
## Deep Learning: Stochastic Gradient Descent

::: columns

:::: {.column width=40%}

Two extremes:

- Full (batch) gradient descent:
$$ \mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L} $$
    - where $\mathcal{L} = \frac{1}{N} {\sum_{i=1}^{N} \mathcal{L}_i}$:  a single update step after computing full gradient $\nabla_\mathbf{W} \mathcal{L}$ over whole dataset!
    - 1 **epoch** - 1 update step !

- Stochastic gradient descent (**SGD**):
$$ \mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_i $$
    - performing update step after computing gradient from a single data point $X_i \in \mathcal{D}$
    - "noisy" gradient estimate $\nabla_\mathbf{W} \mathcal{L}_i$
    - 1 **epoch**: $N$ update steps visiting all $X^i \in \mathcal{D}$    

::::

:::: {.column width=40%}

::::

:::

-->

## Deep Learning: Stochastic Gradient Descent

Two extremes:

- full (batch) gradient descent:
$$ \mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L} $$
    - where $\mathcal{L} = \frac{1}{N} {\sum_{i=1}^{N} \mathcal{L}_i}$: a single update step after computing the full gradient $\nabla_\mathbf{W} \mathcal{L}$ over the whole dataset!
    - 1 **epoch** – 1 update step!

- stochastic gradient descent (**SGD**):
$$ \mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_i $$
    - performing update step after computing gradient from a single data point $X_i \in \mathcal{D}$
    - "noisy" gradient estimate $\nabla_\mathbf{W} \mathcal{L}_i$
    - 1 **epoch**: $N$ update steps visiting all $X^i \in \mathcal{D}$

\center{\includegraphics[width=0.6\textwidth]{../images/Full_SGD_Gradient_Descent_Walk.pdf}}

<!-- * Two extremes:
  - full batch GD : visit whole dataset before doing a single weight update step via full gradient
  - Stochastic GD (SGD): make an update step with gradient from single example -->

## Deep Learning: Stochastic Gradient Descent

Inbetween two extremes: **mini-batch SGD**

- Perform an update step using loss gradient $\nabla_\mathbf{W} \mathcal{L}_B$ over a **mini-batch** of size $\vert B \vert = n \ll N$

$$ \nabla_\mathbf{W} \mathcal{L}_B = \nabla_\mathbf{W} \dfrac{1}{n} {\displaystyle \sum_{i=1}^{n} \mathcal{L}_i} = \dfrac{1}{n} {\displaystyle \sum_{i=1}^{n} \nabla_\mathbf{W} \mathcal{L}_i} $$

- Combines benefits of both extremes:
  - can use **vectorization** for efficient gradient computation; affordable memory demand
  - is still noisy – escaping "bad" loss landscape regions; good for generalization? - still debated
  - fewer steps to converge than pure SGD

\center{\includegraphics[width=0.6\textwidth]{../images/Full_SGD_Gradient_Descent_Walk.pdf}}

## Deep Learning: Mini-Batch SGD

::: block
### Training via mini-batch SGD

* Learning dynamics $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$
  - convergence speed
  - final loss region

* Strongly dependent on
  - **learning rate** $\eta$
  - **batch size** $\vert B \vert$

:::

\center{\includegraphics[width=0.6\textwidth]{../images/Dynamics_SGD_Learning_Cases_Rate.pdf}}

^[\tiny Figure Roger Grosse, U Toronto, CSC421]

## Deep Learning: Mini-Batch SGD

::: block
### Training via mini-batch SGD

* Learning dynamics $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$
  - convergence speed
  - final loss region

* Strongly dependent on many other factors:
  - **network architecture**, **weight initialization**, **input \& activity normalization**, **regularization** (weight decay, ...), **data augmentation**, **optimizer type** (e.g. SGD with Nesterov momentum, ...)

:::

\center{\includegraphics[width=0.6\textwidth]{../images/Dynamics_SGD_Learning_Cases_Rate.pdf}}

^[\tiny Figure Roger Grosse, U Toronto, CSC421]

## Deep Learning: Weight Update Dynamics

::: block
### Training via mini-batch SGD

* Learning dynamics $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$
  - convergence speed
  - final loss region

* Strongly dependent on
  - **learning rate** $\eta$
  - batch size $\vert B \vert$

:::

\center{\includegraphics[width=0.7\textwidth]{../images/Learning_Rate_Loss_Curves.png}}

## Deep Learning: Weight Update Dynamics

::: block
### Training via mini-batch SGD

* Learning dynamics $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$
  - convergence speed
  - final loss region

* Strongly dependent on
  - **learning rate** $\eta$ : rate schedules
  - batch size $\vert B \vert$

:::

\center{\includegraphics[width=0.6\textwidth]{../images/ResNet_CIFAR_ImageNet_LearningRate_Schedule.pdf}}

^[\tiny He et al, CVPR, 2016]

## Deep Learning: Weight Update Dynamics

::: block
### Training via mini-batch SGD

* Learning dynamics $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$
  - convergence speed
  - final loss region

* Strongly dependent on
  - **learning rate** $\eta$ : rate schedules
  - batch size $\vert B \vert$

:::

\center{\includegraphics[width=\textwidth]{../images/Learning_Rate_Schedules.png}}

## Deep Learning: Weight Update Dynamics

::: block
### Training via mini-batch SGD

* Learning dynamics $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$
  - convergence speed
  - final loss region

* Strongly dependent on
  - learning rate $\eta$
  - **batch size** $\vert B \vert$ (changing $\vert B \vert$ often requires changing $\eta$)

:::

\center{\includegraphics[width=0.8\textwidth]{../images/Batch_Size_Dependency_Basic_ImageNet.pdf}}

^[\tiny Goyal et al., 2017]

## Deep Learning: Weight Update Dynamics

::: block
### Training via mini-batch SGD

* Learning dynamics $\mathbf{W}_{t+1} \leftarrow \mathbf{W}_t - \eta \nabla_\mathbf{W} \mathcal{L}_B$
  - convergence speed
  - final loss region

* Strongly dependent on
  - learning rate $\eta$
  - **batch size** $\vert B \vert$: SGD type dependent (Nesterov momentum vs plain SGD, ...)

:::

\center{\includegraphics[width=0.8\textwidth]{../images/Batch_Size_Dependency_Basic_ImageNet.pdf}}

^[\tiny Goyal et al., 2017]

<!--
## Deep Learning: Stochastic Gradient Descent
* Mini-Batch SGD optimization critically depends on
  - weight initialization
  - input and activity normalization
  - learning rate $\eta$
  - batch size $B$

- Different SGD types: plain, (Nesterov) Momentum, RMSProp, ADAM, ...
-->

## Deep Learning: Recap

::: block
### Mini-batch SGD on large data

- Minimizing loss $\mathcal{L}^{tr}$ on very large $\mathcal{D}^{tr}$
- Adapting differentiable complex $f(X;\mathbf{W})$, $\mathbf{W}$ very large
- Using mini-batch loss gradient $\nabla_\mathbf{W} \mathcal{L}_B$ for weight updates
- Loss landscape complex and unknown, many pitfalls
- Optimization critically depends on proper initialization and hyperparams
  - interaction of **learning rate** $\eta$ and **batch size** $\vert B \vert$
    - **batch size** issues: relevant for distributed training
- **Remember**: minimizing $\mathcal{L}^{tr}$ is not main target
  - generalization matters most: estimate gap on $\mathcal{L}^{ts}$

:::

\center{\includegraphics[width=0.7\textwidth]{../images/Basic_ML_Optimization_Data.pdf}}
