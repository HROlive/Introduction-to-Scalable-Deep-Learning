---
author: Mehdi Cherti
title: Advanced Generative Adverserial Networks (GANs)
date: February 5th, 2021
---

# Generative Models

- Impressive progress in last years, algorithmic improvements coupled with **large scale training** and **large models**

\center{\includegraphics[width=0.7\textwidth]{images/gan_evolution.jpeg}}
(Source: <https://bit.ly/3azTV7J>)

# Generative modeling benefit from scaling: BigGAN

- BigGAN [@brock2019large] was the first architecture to scale to ImageNet-1K
- Trained on high resolution up to 512x512, and have **high diversity and high quality samples**

\center{\includegraphics[width=0.8\textwidth]{images/biggan.png}}
 
# Generative modeling benefit from scaling: BigGAN

- Model much larger than previous works: scaling width and and depth
- Batch size (up to 2048) much bigger than previous works
- Benefit of scaling the model: reach better performance in **fewer iterations**

\center{\includegraphics[width=0.8\textwidth]{images/biggan_results.png}}
\center{\includegraphics[width=0.8\textwidth]{images/biggan_results_2.png}}

# Generative modeling benefit from scaling: BigGAN

- 512x512 resolution model trained on 512 TPUs (TPU v3 pod)
- Training takes between 24 hours and 48 hours for most models
- Results as model size is increased:
\center{\includegraphics[width=0.40\textwidth]{images/biggan_results_6.png}}

# Generative modeling benefit from scaling: BigGAN

- a lot of tuning is necessary in experimentation, before finding the good range of hyper-parameters

\center{\includegraphics[width=0.6\textwidth]{images/biggan_results_5.png}}

# Representation learning benefit from scaling: BigBiGAN

- BigBiGAN[@donahue2019large] asked the following question: can GANs learn a useful general representation
from unlabeled data ?
- Can we learn high level concepts that we can exploit for downstream tasks ?

# Representation learning benefit from scaling: BigBiGAN

- Similar to BigGAN but in addition to discriminator and generator, we also have an **encoder**
- Three networks to optimize simultanously

\center{\includegraphics[width=0.8\textwidth]{images/bigbigan_1.png}}

# Representation learning benefit from scaling: BigBiGAN

- Training on ImageNet-1K up to 256x256 resolution, completely unsupervised (no conditioning)
- Training on 32 to 512 TPU cores
- Batch size of 2048 similar to BigGAN
- Architecture similar to BigGAN for **generator** and **discriminator**, 
for **encoder** architecture is based  on ResNet-50
 
\center{\includegraphics[width=0.4\textwidth]{images/bigbigan_5.png}}

# Representation learning benefit from scaling: BigBiGAN

- Learned representation focus on high-level semantic details

\center{\includegraphics[width=0.6\textwidth]{images/bigbigan_3.png}}

\center{\includegraphics[width=0.6\textwidth]{images/bigbigan_4.png}}

# Representation learning benefit from scaling: BigBiGAN

- Better image modeling (FID) translates to better performance in downstream task (supervised classification)
- Bigger models perform **better** in downstream task (supervised classification)

\center{\includegraphics[width=0.34\textwidth]{images/bigbigan_2.png}}

# Innovations in architecture: StyleGAN2

- StyleGAN/StyleGAN2 [@karras2020training] introduces a novel way to structure the generator architecture
- It decomposes the latent space into **high level attributes**
(encoding concepts such pose and identity) and **stochastic
variation** (e.g., to handle freckles and hair)

\center{\includegraphics[width=0.8\textwidth]{images/stylegan_representation.png}}

# Innovations in architecture: StyleGAN2

- The latent $\boldsymbol{z}$ is converted to $\boldsymbol{w} = f(\boldsymbol{z})$
using 8 fully connected layers $f$
- The $\boldsymbol{w}$ vector is then mapped to a style vector using a fully connected network for each resolution. One style vector per resolution.

\center{\includegraphics[width=0.13\textwidth]{images/stylegan_latent_space.png}}

# Innovations in architecture: StyleGAN2

- One block per resolution, each block upsample by 2
- Each resolution block is affected by its dedicated style vector $A$ and noise $B$

\center{\includegraphics[width=0.4\textwidth]{images/stylegan2.png}}

# Innovations in architecture: StyleGAN2

- No need progressive generation like in ProGAN [@karras2018progressive]
- The generated image RGBs are a sum of RGBs from each resolution outputs, everything is learned simultanously

\center{\includegraphics[width=0.7\textwidth]{images/stylegan_output.png}}

# Innovations in architecture: StyleGAN2

- Larger configuration network renders high resolution details better

\center{\includegraphics[width=0.8\textwidth]{images/stylegan_large.png}}

# Innovations in architecture: StyleGAN2

- Distributed training on 8 V100 GPUs
- Reduce training time from 70 days with 1 GPU to 10 days with 8 GPUs

\center{\includegraphics[width=0.8\textwidth]{images/stylegan_time.png}}

# Innovations in architecture: StyleGAN2

- One should not forget the cost of exploration as well: 51 GPU years was required
in total

\center{\includegraphics[width=0.8\textwidth]{images/stylegan_gpuyears.png}}


# StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets

- Up until now, StyleGAN models had difficulties with large diverse datasets such as ImageNet
- By combining different techniques, StyleGAN-XL [@sauer2022stylegan] could achieve state of the art results on ImageNet
for the first time

\center{\includegraphics[width=0.8\textwidth]{images/stylegan-xl.png}}

# StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets

- They leverage several recent techniques to improve sample quality
- In particular, they exploit the rich representation  of several pre-trained models (supervised and self-supervised)

\center{\includegraphics[width=0.8\textwidth]{images/stylegan-xl-ablation.png}}

# Innovations in architecture: GANsFormer

- StyleGAN2 have been shown to have difficulties with datasets with a lot of diversity, e.g., complex scenes with multiple objects
- This is possibly attributed to the fact that one global latent controls all the styles simultanously
- GANsFormer [@hudson2021generative] is a new architecture, based on StyleGAN2, where they have multiple latents and use transformers to integrate information from the latents into the image


# Innovations in architecture: GANsFormer

- we have multiple latents instead of a single one that globally controls the image

\center{\includegraphics[width=0.33\textwidth]{images/gansformer1.png}}

# Innovations in architecture: GANsFormer

- To integrate information from the latents $Y$ into the image $X$, they use a transformer architecture
- To make the transformer efficient, they use a bipartite structure, where connections are made between image features and latents only

\center{\includegraphics[width=0.6\textwidth]{images/gansformer6.png}}

# Innovations in architecture: GANsFormer

- Different latents specialize in different aspects of the image

\center{\includegraphics[width=0.75\textwidth]{images/gansformer3.png}}

# Summary

- We have seen different architectures proposed in the literature
- The GANs are in general costly to train, especially with larger
resolutions and for large datasets. 
- Distributed training helps to make training faster

# References {.allowframebreaks}
