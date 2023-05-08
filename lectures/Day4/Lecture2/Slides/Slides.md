<!--- This is how a comment in Markdown looks like -->

## Large-scale Pre-Training and Transfer

- Foundation models: transferable models pre-trained on large generic data
  - transfer across domains specific smaller datasets and tasks
  - scaling laws: strong, efficient transfer - large models pre-trained on large data

\center{\includegraphics[width=0.8\textwidth]{../images/Pretrain_Transfer_Foundation_Model_mod.pdf}}

## Large-scale Pre-Training and Transfer

- Supervised learning on generic images : relating images to low information labels
<!-- - Supervised learning: relating high information input signal with low information output label -->

\center{\includegraphics[width=0.8\textwidth]{../images/Supervised_Learning_Car_Image.pdf}}

## Large-scale Pre-Training and Transfer

- Supervised learning : relating high information input signal to low information labels
  - fixed, rather small label "vocabulary"

\center{\includegraphics[width=0.8\textwidth]{../images/Supervised_Learning_Labels_Teacher_mod.pdf}}


## Large-scale Pre-Training and Transfer

- Scaling Laws: Larger Data, Larger Models - better transfer
- Supervised learning on ImageNet-1k : pre-trained models transferable
  - pre-train on ImageNet-1k - transfer across various downstream tasks
- Problem: Human-labeled data poorly scalable
  - **ImageNet-21k**: **14x** larger
  - **JFT-300M**: **300x** larger - pseudo-labels

\center{\includegraphics[width=0.6\textwidth]{../images/Supervised_Learning_Car_Image.pdf}}


## Large-scale Pre-Training and Transfer

- Scaling Laws: Larger Data, Larger Models - better transfer
- Supervised learning on ImageNet-1k : pre-trained models transferable
  - pre-train on ImageNet-1k - transfer across various downstream tasks
- Problem: poor zero-shot transfer, poor robustness to data distribution shift
  - **ImageNet-21k**: **14x** larger
  - **JFT-300M**: **300x** larger - pseudo-labels

\center{\includegraphics[width=0.6\textwidth]{../images/Supervised_Learning_Car_Image.pdf}}

## Large-scale Pre-Training and Transfer

- Scaling Laws: Larger Data, Larger Models - better transfer
- Supervised learning on ImageNet-1k : pre-trained models transferable
  - pre-train on ImageNet-1k - transfer across various downstream tasks
- Problem: poor zero-shot transfer, poor robustness to data distribution shift
  - **ImageNet-21k**: **14x** larger
  - **JFT-300M**: **300x** larger - pseudo-labels

\center{\includegraphics[width=\textwidth]{../images/Pretraining_BiT_Data_Networks_smaller_larger.png}}

^[\tiny BiT - Big Transfer, Kolesnikov et al, ECCV, 2020]

## Large-scale Pre-Training and Transfer
- Scalable data: unlabeled or pseudo-labeled data
- **Unsupervised**, **Self-Supervised** learning in different flavors
  - human-made labels not required
- Often, using auxiliary tasks - self-supervised learning
  - contrastive losses (SimCLR, DINO), reconstruction based losses (eg VAEs, MAE, Diffusion models), ...
  - adversarial losses (eg. GANs -> see Day 5 Special!)

\center{\includegraphics[width=\textwidth]{../images/unsupervised_learning_overview.png}}

^[\tiny Pidhorskyi et al, 2020; Effenberger et al, 2020; Chen et al, 2020]

## Large-scale Pre-Training and Transfer
- Scalable data: unlabeled data
- Contrastive losses: construct losses from transformed pairs of inputs

::: columns

:::: {.column width=50%}

\small
$$
\mathbf{z}_i = g(\mathbf{h}_i),\quad
\mathbf{z}_j = g(\mathbf{h}_j),\quad
\text{sim}(\mathbf{z}_i, \mathbf{z}_j) = \frac{\mathbf{z}_i^\top\mathbf{z}_j}{\|\mathbf{z}_i\| \|\mathbf{z}_j\|}
$$

\vspace*{-0.2cm}
$$
\mathcal{L}_{i,j} = - \log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2n} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
$$

\normalsize

\center{\includegraphics[width=0.7\textwidth]{../images/SimCLR_plane.png}}


::::

:::: {.column width=50%}

\center{\includegraphics[width=0.7\textwidth]{../images/SimCLR_Attract_Repell.png}}

::::

:::

^[\tiny Chen et al, 2020]

<!-- \mathbf{h}_i = f(\tilde{\mathbf{x}}_i),\quad \mathbf{h}_j = f(\tilde{\mathbf{x}}_j) -->


## Large-scale Pre-Training and Transfer
- Scalable data: unlabeled data
- Contrastive losses: larger models - better self-supervised learning
- Evidence for better representations in larger networks after self-supervised pre-training  

\center{\includegraphics[width=0.8\textwidth]{../images/SimCLR_Linear_Classifier_Slide.png}}

^[\tiny Chen et al, 2020]

## Large-scale Pre-Training and Transfer
- Scalable data: weakly aligned image-text pairs from public Internet
- CLIP: foundation language-vision model for large-scale representation learning
  - **self-supervised, open vocabulary pre-training on image-text pairs**
  - very strong zero- and few-shot transfer across various downstream tasks
  - strong robustness to data distribution shift

\center{\includegraphics[width=0.85\textwidth]{../images/CLIP_Self_Supervised_Learning_mod.pdf}}

<!--\vspace{-0.5cm}-->
^[\tiny Radford, ICML, 2021]

## Large-scale Pre-Training and Transfer
- Scalable data: weakly aligned image-text pairs from public Internet
- CLIP: foundation language-vision model for large-scale representation learning
  - self-supervised, open vocabulary pre-training on image-text pairs
  - **very strong zero- and few-shot transfer across various downstream tasks**
  - strong robustness to data distribution shift

\center{\includegraphics[width=0.8\textwidth]{../images/CLIP_Zero_Shot_Few_Shot_Scaling_mod.pdf}}

<!--\vspace{-0.5cm}-->
^[\tiny Radford, ICML, 2021; Schumann et al, NeurIPS 2022]


## Large-scale Pre-Training and Transfer
- Scalable data: weakly aligned image-text pairs from public Internet
- CLIP: foundation language-vision model for large-scale representation learning
  - self-supervised, open vocabulary pre-training on image-text pairs
  - very strong zero- and few-shot transfer across various downstream tasks
  - **strong robustness to data distribution shift**

\center{\includegraphics[width=0.85\textwidth]{../images/CLIP_Effective_Robustness_Dataset_Shift_mod.pdf}}

<!--\vspace{-0.5cm}-->
^[\tiny Radford, ICML, 2021]

## Large-scale Pre-Training and Transfer
- CLIP - language-vision foundation model; self-supervised language-vision learning (no labels)
- Out-of-distribution robustness & few-shot / zero-shot transfer
- Pre-trained models are **highly re-usable across various tasks & conditions**
- Generalist zero-shot function: no adaptation to new conditions / data / tasks required

\center{\includegraphics[width=0.9\textwidth]{../images/CLIP_re_usable_component_mod.pdf}}

^[\tiny Rombach et al, CVPR, 2022; Khandelwal et al, CVPR, 2022]



## Large-scale Pre-Training and Transfer
- Problem - studying self-supervised foundation models is challenging: requires
  - large-scale **data** (at least 100M of samples)
  - large-scale **compute** (in order of GPU months per single experiment)
  - **expertise** in large-scale distributed training

\center{\includegraphics[width=0.2\textwidth]{../images/LAION_logo_mod.pdf}}

- Solution - LAION: Large-scale Artificial Intelligence Open Network
  - **data**: LAION-400M, LAION-5B image-text datasets – **Outstanding Paper Award NeurIPS 2022**
  - **compute**: applying for publicly funded supercomputers (JUWELS, Germany, SUMMIT, USA)
  - **expertise**: strong grassroot research community skilled in large-scale experiments and distributed training
  - **Open-source** release of pre-trained models: openCLIP (work published at NeurIPS, CVPR)

## Large-scale Pre-Training and Transfer
- LAION-400m/5B (2021): next gen datasets, 10x/100x larger than ImageNet-1k/21k, multi-modal (image-text)
  - data collection from public Internet (Common Crawl) by community effort (http://laion.ai)
- CommonPool-10B, DataComp-1B: follow-up work; systematic dataset search for pre-training strong models

\center{\includegraphics[width=0.8\textwidth]{../images/ImageNet_LAION_Transition_mod.pdf}}

^[\tiny  Schumann et al, NeurIPS 2022; Gadre, Ilharco, Fang et al, arXiv:2304.14108, 2023]

## Large-scale Pre-Training and Transfer
- Open data LAION-400M/5B, DataComp: Open sourcing data collection procedures
  - transparent dataset, open source tools \& workflows, reproducible training across various scales
  - dataset of links to images in public internet, together with text captions
  - researchers can obtain full image-text dataset for experiments using open tools

\center{\includegraphics[width=0.9\textwidth]{../images/LAION_Dataset_mod.pdf}}

^[\tiny  Schumann et al, NeurIPS 2022; Gadre, Ilharco, Fang et al, arXiv:2304.14108, 2023]

## Large-scale Pre-Training and Transfer
- Open-source foundation data and models - reproducible, open science

\center{\includegraphics[width=0.7\textwidth]{../images/openCLIP_reproducible_LAION_comparison_plot_openAI_mod.pdf}}

^[\tiny  Schumann et al, NeurIPS 2022]

## Large-scale Pre-Training and Transfer
- Scaling laws for language-vision learning with LAION and openCLIP: open-source data, models and code - reproducible science

\center{\includegraphics[width=\textwidth]{../images/LAION_openCLIP_scaling_zero_shot_mod.pdf}}

^[\tiny  Schumann et al, NeurIPS 2022]


## Large-scale Pre-Training and Transfer
- Scaling laws for language-vision learning with LAION and openCLIP: open-source data, models and code - reproducible science

\center{\includegraphics[width=0.9\textwidth]{../images/openCLIP_reproducible_LAION_table_mod.pdf}}

^[\tiny  Schumann et al, NeurIPS 2022]


## Large-scale Pre-Training and Transfer
- Bottlenecks: one insufficient scale can lead to saturation if increasing others
- Larger language-vision models: stronger on larger dataset and sample seen scales

\center{\includegraphics[width=0.9\textwidth]{../images/openCLIP_reproducible_LAION_scales_bottleneck_table_mod.pdf}}

^[\tiny Cherti et al, arXiv:2212.07143, CVPR 2023]

## Large-scale Pre-Training and Transfer
- Scaling laws for language-vision learning with LAION and openCLIP: open-source data, models and code - reproducible science
- Predicting model performance and properties on larger scales

\center{\includegraphics[width=\textwidth]{../images/openCLIP_reproducible_LAION_scaling_law_prediction_mod.pdf}}

^[\tiny Cherti et al, arXiv:2212.07143, CVPR 2023]

## Large-scale Pre-Training and Transfer
- Scaling laws for language-vision learning with LAION and openCLIP: open-source data, models and code - reproducible science
- Predicting model performance and properties on larger scales

\center{\includegraphics[width=0.9\textwidth]{../images/openCLIP_reproducible_LAION_scaling_law_prediction_table_mod.pdf}}

^[\tiny Cherti et al, arXiv:2212.07143, CVPR 2023]


## Large-scale Pre-Training and Transfer
- JUWELS Booster: necessary for the experiments
- 122 hours with 1024 A100 (124K GPU hours) for training of ViT L/14 openCLIP on 34B samples
- In contrast to standard supervised training: larger batch sizes beneficial for learning

\center{\includegraphics[width=\textwidth]{../images/openCLIP_scaling_efficiency_plots_L14_mod.pdf}}

^[\tiny Cherti et al, arXiv:2212.07143, CVPR 2023]

## Large-scale Pre-Training and Transfer
- Current language-vision models are still small scale (compared to LLMs (>100B params); PaLI (image-text-to-text) – 17B; Parti (text-to-image)  - 20B params)
- Stronger transfer \& robustness: aiming for larger scales
- Larger machines necessary: JUPITER Exascale upcoming at JSC

\center{\includegraphics[width=0.9\textwidth]{../images/openCLIP_GMAC_params_table_mod.pdf}}

^[\tiny Cherti et al, arXiv:2212.07143, CVPR 2023]

## Distributed Training with Very Large Models
- Growing model scale: only data parallel scheme not sufficient
  - Language Modelling: GPT-3 - 175 Billion parameters; PaLM (Google) - 540B parameters
  - Vision: ViT-22B; Language-Vision: Parti - 17B params
- Model/Tensor parallelism, Pipeline Parallelism: can split a very large model across accelerators
- Different libraries: DeepSpeed (Microsoft), CollosalAI (HPC-AI), PyTorch/TensorFlow DTensor, ...

\center{\includegraphics[width=\textwidth]{../images/Distributed_Training_Schemes_Titled_Slides.pdf}}

^[\tiny Laskin et al, 2020]


## Distributed Training with Very Large Models
- Hybrid parallel schemes
  - using data, model and pipeline parallelism simultaneously
- Distributed training that combines memory and compute efficiency
- DeepSpeed: supports hybrid parallelism

\center{\includegraphics[width=0.7\textwidth]{../images/DeepSpeed_Hybrid_Parallel_Slide.pdf}}

## Distributed Training with Very Large Models
- Hybrid parallel schemes
  - using data, model and pipeline parallelism simultaneously
- DeepSpeed: "3D Parallelism"
  - executing and speeding up a Trillion size model on 800 A100 GPUs

\center{\includegraphics[width=0.8\textwidth]{../images/DeepSpeed_Scaling_Slide.pdf}}

<!-- "3D Parallelism": 8-way data, 8-way model and 64-way pipeline parallelism
4096 GPUs ... -->


## Distributed Training with Very Large Models
- Upcoming: local updates, decoupled gradients
- Getting rid of global forward-backward pass dependency alltogether
- Asynchronous local updates, highly beneficial for parallelization
- Towards energy-efficient in-memory computing: minimize data transfer
- New generic losses for unsupervised learning

\center{\includegraphics[width=0.5\textwidth]{../images/Local_Losses_Decoupled_Slide.png}}

^[\tiny Laskin et al, 2020]

## Distributed Training with Very Large Models
- Upcoming: local updates, decoupled gradients
- Asynchronous local updates, highly beneficial for parallelization
- Energy efficient distributed training on specialized hardware, in-memory computing
  - Graphcore IPU: Colossus Mk2
  - Cerebras : Wafer Scale Engine 2 (WSE - 850k Cores!)

\center{\includegraphics[width=\textwidth]{../images/Cerebras_Local_Losses_Slide.png}}

<!--
## Distributed Training: Beginning of A Journey
- Large Scale Learning in Simulated Environments
  - **Distributed Generative Active Learning**: Data Selection and Generation in the Loop
  - **Differentiable simulators** integrated into learning loop - physics-based regularization and learning
- Modular Supercomputing containing different accelerator types
  - Modular Supercomputers are designed at JSC

\center{\includegraphics[width=\textwidth]{../images/Coupled_Simulators_Distributed_RL_Slide.pdf}}

^[\tiny Jadeberg et al, Science, 2019]
-->

## Large-Scale Foundation Generalist Models

::: block

### Outlook

* Language-vision generalist models for strong transfer across domains and tasks
* Large model scale: hybrid parallelism required
* Large data scale: data collection and automated filtering (see DataComp)
* Systematic Search for Scalable Architectures (Project Nucleus, LAION)
* Model compression for efficient transfer and low resource deployment
* Energy efficient large scale learning with in-memory computing neuromorphic hardware

:::


\center{\includegraphics[width=0.6\textwidth]{../images/JUWELS_Booster_Slide.pdf}}


<!--

* Distributed generative active learning: simulators in the learning loop

* Continual large-scale learning
* Modular Supercomputers

-->

## Large-Scale Foundation Generalist Models
- LAION: Large-Scale Artificial Intelligence Open Network (join on Discord!)
  - Scalable Learning \& Multi-Purpose Lab (SLAMPAI; Jenia Jitsev, Mehdi Cherti)
  - University of Washington (Seattle), Allen AI Institute, MILA, UC Berkeley, U Tel-Aviv, Stanford, ...
  - https://laion.ai/ - join on Discord!
- Supercomputers : JUWELS \& JUWELS Booster, JUPITER to come

\center{\includegraphics[width=0.8\textwidth]{../images/LAION_JUWELS.png}}

## Large-Scale Foundation Generalist Models
- WestAI: AI Service Center, funded by BMBF (2022-2025, 12.4M €)
- Pillars: large-scale pre-training, scaling laws for transfer, compression, next-gen highly scalable generic, energy-efficient learning
  - SLAMPAI at JSC: scientific lead
  - U Bonn, RWTH Aachen, Fraunhofer IAIS, U Padeborn, U Dortmund

\center{\includegraphics[width=0.9\textwidth]{../images/WestAI_Consortium.png}}

## Distributed Training: Activities at FZJ
- Distributed Training for Hyperspectral Remote Sensing (Gabriele Cavallaro)
- Helmholtz Data Challenges : Platform for collaborative datasets and model training
  - SLAMPAI \& Helmholtz AI Consultant Team: https://helmholtz-data-challenges.de/
- TOAR: Earth System Data Exploration (ESDE) Lab (Martin Schultz)
- Helmholtz AI Research Group at INM-1: deep learning for neuroimaging
- JULAIN: Juelich Artificial Intelligence Network, join in!
  - mailing list: https://lists.fz-juelich.de/mailman/listinfo/ml

\center{\includegraphics[width=0.6\textwidth]{../images/JUWELS_Booster_Slide.pdf}}

## Large-Scale Foundation Generalist Models
- LAION: Large-Scale Artificial Intelligence Open Network (join on Discord!)
  - Scalable Learning \& Multi-Purpose Lab (SLAMPAI; Jenia Jitsev, Mehdi Cherti)
  - University of Washington (Seattle), Allen AI Institute, MILA, UC Berkeley, U Tel-Aviv, Stanford, ...
  - https://laion.ai/ - join on Discord!
- Supercomputers : JUWELS \& JUWELS Booster, JUPITER to come

\center{\includegraphics[width=0.8\textwidth]{../images/LAION_JUWELS.png}}
