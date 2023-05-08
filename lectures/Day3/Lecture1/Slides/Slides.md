<!--- This is how a comment in Markdown looks like -->


## Large Networks, Large Datasets

- Training models that solve complex, real world tasks requires large model and data scale

\vspace{-0.2cm}
\center{\includegraphics[width=0.9\textwidth]{../images/Compute_Budget_Training_Milestone_models_2022_mod.pdf}}

\vspace{-0.5cm}
<!--^[\tiny Amodei et al, 2019 (Source: https://openai.com/blog/ai-and-compute/)]
-->
^[\tiny Sevilla et al., 2023; OurWorldInData.org/artificial-intelligence; CC BY Source]

<!--\center{\includegraphics[width=0.7\textwidth]{../images/OpenAI_Modern_Compute.pdf}}
-->

## Large Networks, Large Datasets

- Networks : large models, many layers, large number of parameters (weights)
  - Vision: Convolutional, Transformer and Hybrid networks
  - hundreds of layers, hundred millions of parameters (currently up to 20B)

\center{\includegraphics[width=\textwidth]{../images/Vision_Networks_CoAtNet_2021_mod.pdf}}

^[\tiny Dai et al, NeurIPS, 2021]

## Large Networks, Large Datasets

- Networks : large models, many layers, large number of parameters (weights)
  - Language: Transformer networks
  - hundreds of layers, billions of parameters (GPT-3: 175 Billion)

\center{\includegraphics[width=0.65\textwidth]{../images/MegaTron_DeepSpeed_Microsoft_Turing_NLM_model-size-graph.jpg}}

\vspace*{-1cm}
^[\tiny Narayanan et al, 2021]

<!--
## Large Networks, Large Datasets

- Millions, even Billions of network parameters: training demands data
- Most breakthroughs happened on large data
  - Vision: ImageNet-1k (1.4 M images); ImageNet-21k (14 M images, $\approx$ 4 TB uncompressed)
  - Language: LM1B, 1 Billion Word Language Model Benchmark
- Datasets get larger and larger  
  - JFT-300 (300 M images); YouTube-8M, 8 Million videos, 300 TB
  - Common Crawl dataset : 280 TB uncompressed text, ca. trillion words (as of 2020)

\center{\includegraphics[width=0.6\textwidth]{../images/ImageNet_Outperforming.pdf}}
-->

## Large Networks, Large Datasets
<!-- - ResNet, DenseNet, EfficientNet, Transformers: Millions, even Billions of parameters -->
- Millions, Billions of network parameters: training demands data
- Most breakthroughs happened on large data; datasets for model training get larger and larger
  - Vision, Supervised, Self-Supervised: ImageNet-1k (1.4 M images); ImageNet-21k (14 M images, $\approx$ 1.4 TB compressed); JFT-300M/4B (300M/4B images); YouTube-8M (8 Million videos, 300 TB)
  - Language-Vision, Self-Supervised
    - CLIP trained on WIT-400M (400M image-text pairs); openCLIP on LAION-400M/5B (open data, 400M/5B image-text pairs, 11TB/240TB)
    - Stable Diffusion trained on LAION-5B
  - Language, Self-supervised
    - GPT-3 trained on 300-400 Billion word tokens
    - LLaMA, RedPajama (open) : 5 TB uncompressed text, ca. 1.2 trillion tokens

\center{\includegraphics[width=0.8\textwidth]{../images/Datasets_Table_Overview_Small_Large_mod.pdf}}

## Reconciling Large Models and Generalization
<!-- - ResNet, DenseNet, EfficientNet, Transformers: Millions, even Billions of parameters -->
- Both network models and datasets get larger and will continue to grow
  - Generalization: large models and the generalization gap

\center{\includegraphics[width=\textwidth]{../images/Generalization_Training_test_Error.pdf}}

^[\tiny Goodfellow et al, 2016]

## Reconciling Large Models and Generalization
<!-- - ResNet, DenseNet, EfficientNet, Transformers: Millions, even Billions of parameters -->
- A (classical) simple view - more data, better generalization

\center{\includegraphics[width=\textwidth]{../images/Simple_Story_More_Data_Toy_Bishop.png}}

^[\tiny Bishop, 2006]

## Reconciling Large Models and Generalization
<!-- - ResNet, DenseNet, EfficientNet, Transformers: Millions, even Billions of parameters -->
- A (classical) simple view - more data, better generalization
  - Never enough data in higher dimensions - curse of dimensionality

\center{\includegraphics[width=\textwidth]{../images/Simple_Story_More_Data_Toy_Bishop.png}}

^[\tiny Bishop, 2006]

<!--
## Reconciling Large Models and Generalization

- A (very recent) complex view - larger models, better generalization

\center{\includegraphics[width=\textwidth]{../images/Generalization_Training_test_Error.pdf}}

^[\tiny Goodfellow et al, 2016]
-->


## Reconciling Large Models and Generalization

- A (very recent) complex view - larger models, better generalization
  - **Double descent** test error curve, going beyond **interpolation threshold**
  - Greatly increasing number of model parameters **reduces** generalization gap

\center{\includegraphics[width=\textwidth]{../images/Conventional_Reconciled_Generalization_View_Belkin_PNAS.pdf}}

^[\tiny Belkin et al, PNAS, 2019]

<!-- generalization picture simple reason more data better model for large models  -->

## Reconciling Generalization Gap

\center{\includegraphics[width=0.8\textwidth]{../images/MNIST_CIFAR_Generalization_Reconciling_Belkin_Test.pdf}}

^[\tiny Belkin et al, PNAS, 2019]

<!-- Way Road free for scaling up -->

## Reconciling Large Models and Generalization

- Larger models generalize better
  - Reconciling generalization - large, overparameterized models generalize strongly
  - Greatly increasing number of model parameters **reduces** generalization gap
  - **Double descent** test error curves

<!-- \center{\includegraphics[width=\textwidth]{../images/Deep_Double_Descent_ResNet_18_CIFAR_10_mod.pdf}} -->
\center{\includegraphics[width=\textwidth]{../images/Double_Descent_Large_Models_mod.pdf}}


^[\tiny Belkin et al, PNAS, 2019, Nakkiran et al, ICLR, 2020]

## Large Models and Generalization

- Larger models generalize better
  - Evidence across different large scale training scenarios

\center{\includegraphics[width=\textwidth]{../images/ResNet_DenseNet_Improving_Model_Size_ImageNet-1k.png}}

^[\tiny Huang et al, CVPR, 2017]

## Large Models and Generalization

- Larger models generalize better
  - Evidence across different large scale training scenarios

\center{\includegraphics[width=\textwidth]{../images/GPT_Model_and_Data_size_Test_error.png}}

^[\tiny Kaplan et al, 2020; Brown et al, NeurIPS, 2020]

## Large Models and Large Data

- Scaling Laws: given sufficient compute budget, increasing both model size and data size is the way to further strongly boost generalization

\center{\includegraphics[width=\textwidth]{../images/Scaling_Compute_Data_Model_Size_Performance_Increase_Kaplan_Test.pdf}}

^[\tiny Kaplan et al, 2020]


## Large Models and Large Data

- Increasing model size is **good** idea, provided enough compute and data

\center{\includegraphics[width=\textwidth]{../images/Scaling_Compute_Data_Model_Size_Performance_Increase_Kaplan_Test.pdf}}

^[\tiny Kaplan et al, 2020]

## Large Models, Data and Generalization
- Language Modeling : very large models, very large data, generic self-supervised pre-training (autoregressive generative sequence models)
  - GPT-3, trained on Common Crawl \& co (300-400B word token samples)
  - Strong few-show and zero-shot transfer at largest scale (175B params)
<!--
  - Image understanding: modest model and data size, supervised pre-training
  - mixed evidence for cross-domain transfer performance
-->

\center{\includegraphics[width=0.7\textwidth]{../images/GPT_Model_size_few_shot_context_performance_mod.pdf}}

^[\tiny Brown et al, NeurIPS, 2020]

## Large Models, Data and Generalization

* Larger models transfer better
  - Evidence across different large scale training scenarios
  - Using large models (BiT - Big Transfer, ResNet-152x4: 928M params), large data
    - ImageNet-21k, $\sim 14$M images (instead of standard ImageNet-1k, $\sim 1.4$M)
    - JFT-300M : $\approx$ 18K classes, noisy labels, 300x larger than ImageNet-1k
  - Pre-training a single large model: **81 hours** with **256 A100** GPUs (20k GPU hours; ImageNet-21k, JUWELS Booster)

\center{\includegraphics[width=\textwidth]{../images/Pretraining_BiT_Data_Networks_smaller_larger.png}}

^[\tiny BiT - Big Transfer, Kolesnikov et al, ECCV, 2020]


## Large Models, Data and Generalization
- Self-supervised language-vision pre-training: GPT for multi-modal image-text data
  - CLIP: very strong zero- and few-shot transfer across various targets
  - very large data for pre-training (eg. open LAION-400M/5B, image-text pairs)
- Larger model and data scale - better zero- and few-shot transfer

\center{\includegraphics[width=0.85\textwidth]{../images/CLIP_Zero_Shot_Few_Shot_Scaling_mod.pdf}}

<!--\vspace{-0.5cm}-->
^[\tiny Radford, ICML, 2021; Schumann et al, NeurIPS 2022]

## Large Models, Data and Generalization
- Larger model and data scale - better zero-shot transfer
  - **88 hours** with **400 A100** (50K GPU hours) for training of ViT L/14 openCLIP on LAION-400M (JUWELS Booster)

\center{\includegraphics[width=\textwidth]{../images/LAION_openCLIP_scaling_zero_shot_mod.pdf}}

^[\tiny Schumann et al, NeurIPS, 2022]

## Large Models, Data and Generalization
::: block

### Summary
- Theoretical insights suggest revision of model generalization at larger scales
  - generalization can improve with larger model scales  
- Scaling laws suggest that larger scales may be one key to strong generalization, model robustness and transferability
- Major breakthroughs in model transferability and robustness in language (GPT) and vision (ViT, CLIP) when using very large model, data and compute scale
- Experiments involving strongly transferable models at larger scale are extremely compute intensive
  - ten or hundred thousands of GPU hours

:::

## Distributed Training with Large Data

- ImageNet: transition to modern deep learning era;
  - outstanding effort in large data collection (Fei-Fei et al, Stanford)
  - building dataset via crowdsourcing over 4 years

\center{\includegraphics[width=0.8\textwidth]{../images/MNIST_CIFAR_ImageNet_Transition_Test.pdf}}


## Distributed Training on ImageNet

- Full dataset (ImageNet-21k) : 14M images, 21k classes labeled
- ImageNet-1k : dataset for ILSVRC competition (2010 - 2017), 1k classes
    - 1.28M Training, 100k Test, 50k Validation sets
    - usual image resolution used for training: 224x224
    - current accuracies : $> 88\%$ top-1,  $>97\%$ top-5  

\center{\includegraphics[width=0.6\textwidth]{../images/ImageNet_Pascal_Transition_Test.pdf}}

^[\tiny Russakovsky et al, IJCV, 2015]

## Distributed Training on ImageNet

- Full dataset (ImageNet-21k) : 14M images, 21k classes labeled
- ImageNet-1k : dataset for ILSVRC competition (2010 - 2017), 1k classes
    - 1.28M Training, 100k Test, 50k Validation sets
    - usual image resolution used for training: 224x224
    - current accuracies : $> 88\%$ top-1,  $>97\%$ top-5  

\center{\includegraphics[width=0.8\textwidth]{../images/ImageNet_Accuracy_Detection_Top_Only_Test.pdf}}

^[\tiny Russakovsky et al, IJCV, 2015]

## Distributed Training on ImageNet

- ImageNet-1k : still gold standard in training large visual recognition models
  - pre-trained models: transfer learning on more specific smaller datasets
- ResNet-50 : baseline model network, accuracies : $\approx 75\%$ top-1,  $\approx 94\%$ top-5 (Winner ILSVRC 2015)

\center{\includegraphics[width=0.8\textwidth]{../images/ImageNet_Accuracy_Detection_Top_Only_Test.pdf}}

^[\tiny Russakovsky et al, IJCV, 2015]

## Distributed Training on ImageNet
- ResNet-50 : efficient distributed training in data parallel mode possible
  - 25M weights, 103Mb for activations, model training on 224x224 ImageNet-1k
  - $\approx 4$ GB Memory with $B_{ref}=64$ : fits onto single GPU

\center{\includegraphics[width=0.6\textwidth]{../images/Yamazaki_RunTime_Table_ImageNet_ResNet-50_Accuracies.png}}

 \center{ \hspace*{5cm} \includegraphics[width=0.4\textwidth]{../images/ResNet-50_ImageNet_Training_Time_Validation_Accuracy.png}}

\vspace*{-1cm}
^[\tiny Yamazoto et al, 2019; Ying, 2018]


## Distributed Training on ImageNet
- Efficient distributed training in data parallel mode
  - requires good scaling of throughput Images/sec during training
  - image throughput during training ideally increasing as $\tau_K^{*} = K \cdot \tau_{ref}$ Images/sec

\center{\includegraphics[width=0.9\textwidth]{../images/JUWELS_Booster_Ims_SpeedUp_ResNet-152x4_mod.pdf}}

^[\tiny Cherti and Jitsev, arXiv:2106.00116, MedNeurIPS Workshop, 2021]

## Distributed Training on ImageNet
- Efficient distributed training in data parallel mode
  - requires good scaling of throughput Images/sec during training

\center{\includegraphics[width=0.9\textwidth]{../images/Data_Parallel_Model_Workers_Processes_Data_Subsets.pdf}}

## Distributed Training on ImageNet

- Efficient distributed training in data parallel mode

::: block

### Data IO
- Efficient file system, efficient data container
    - few separate large files; **sequential access**
    - LMDB, HDF5, TFRecords, WebDataset
- Efficient Data pipeline
    - eg tf.data : interleave, cache, prefetch, ...
    - avoid GPU starvation

:::

```
...

141M /p/largedata/cstdl/ImageNet/imagenet-processed/train-00171-of-01024
137M /p/largedata/cstdl/ImageNet/imagenet-processed/train-00172-of-01024
139M /p/largedata/cstdl/ImageNet/imagenet-processed/train-00173-of-01024
142M /p/largedata/cstdl/ImageNet/imagenet-processed/train-00174-of-01024

...

```

<!--  \center{\includegraphics[width=0.5\textwidth]{../images/Data_Parallel_Model_Workers_Processes_Data_Subsets.pdf}} -->

## Distributed Training on ImageNet
- Efficient distributed training in data parallel mode
  - requires efficient balance of GPU gradient compute and communication    

\center{\includegraphics[width=0.9\textwidth]{../images/Forward_Backward_Data_Parallel_Workers_Batches.pdf}}

## Distributed Training on ImageNet

- Efficient distributed training in data parallel mode

::: block

### SGD Optimization
- Corresponds to training single model with a larger effective batch size $\vert \mathfrak{B} \vert = K \cdot \vert B_{\text{ref}} \vert$
  - Image Throughput ideally increasing as $\tau_K^{\*} = K \cdot \tau_{ref}$ Images/sec
- Make sure model fits into GPU memory
  - remember: this also depends on worker's batch size $\vert B_{\text{ref}} \vert$ and input image resolution
- Avoid internode communication overhead \& bottlenecks
  - Most compute for forward-backward passes
  - $\vert B_{\text{ref}} \vert$ per GPU not too small
  - High capacity network: InfiniBand
  - Horovod: additional mechanisms, eg. Tensor Fusion


:::

<!-- Figure GPU gradient compute and communication -->

## Distributed Training on ImageNet
- ResNet-50 : efficient distributed training in data parallel mode on ImageNet-1k
- Ultimate aim: reducing training **time to accuracy**
  - increasing throughput Images/sec during training only intermediate station!
  - necessary, but not sufficient condition for speeding up model training

\center{\includegraphics[width=0.8\textwidth]{../images/Yamazaki_RunTime_Table_ImageNet_ResNet-50_Accuracies.png}}

\vspace*{-1cm}
^[\tiny Yamazoto et al, 2019]

<!-- -->


## Distributed Training on ImageNet

::: block

### SGD Optimization
- Large effective batch size $\vert \mathfrak{B} \vert$ may require hyperparameter retuning
  - Reminder: Large effective batch sizes alter optimization

:::

\center{\includegraphics[width=0.9\textwidth]{../images/Large_Small_Reference_Batch_Data_Parallel_Model_Workers_Processes.pdf}}

## Distributed Training on ImageNet
- Efficient distributed training in data parallel mode
- Large effective batch sizes may require hyperparameter re-tuning
  - learning rate and schedule
  - optimizer type
- Reminder: hyperparameter tuning for a given $\vert \mathfrak{B} \vert$ - on the validation set!


\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}

^[\tiny Goyal et al, 2017]


## Distributed Training on ImageNet
- Efficient distributed training in data parallel mode
  - Outlook: coping with training on large effective batch sizes
  - Reducing training **time to accuracy**


\center{\includegraphics[width=0.8\textwidth]{../images/Batch_Size_Critical_ImageNet_ResNet-50_Goal_Error.png}}

^[\tiny Shallue et al, JMLR, 2019]

## Large Models, Large Data
::: block

### Summary
- Reconciling generalization: large models generalize better
  - given enough data and compute to train
- Efficient data parallel training on large datasets like ImageNet-1k : possible
- Data pipelines, high bandwidth \& low latency (eg InfiniBand), large batch sizes pave the way
- Implementation of efficient distributed training: Horovod, PyTorch DDP, ... 
- Measures to stabilize training with large batches - upcoming lectures  

:::

\center{\includegraphics[width=0.8\textwidth]{../images/Effective_Reference_Batch_ImageNet_Goyal_2.pdf}}

<!-- \center{\includegraphics[width=0.65\textwidth]{../images/}}
-->
