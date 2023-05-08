---
author: Mehdi Cherti
title: Generative models, Generative Adversarial Networks (GANs) basics
date: February 5th, 2021
---

# Generative Models

- Impressive progress in last years, algorithmic/architectural improvements coupled with large scale training
- Lot of different applications: image generation, text generation, speech synthesis, and more

\center{\includegraphics[width=0.8\textwidth]{images/generated_ffhq_karras2020.png}}
@karras2020training

# Generative Models

- Impressive progress in last years, algorithmic improvements coupled with large scale training
- Lot of different applications: image generation, text generation, speech synthesis, and more

\center{\includegraphics[width=0.68\textwidth]{images/gan_evolution.jpeg}}
(Source: <https://bit.ly/3azTV7J>)

# Generative Models: basics
- The general setup: we have a set of samples $x_1,x_2,...,x_N$ drawn
i.i.d. from an unknown probability distribution $p(x)$. We would like to learn
a model $M$, which we can use to generate samples from $p$
- Different formulations, training algorithms, architectures

\center{\includegraphics[width=0.8\textwidth]{images/generated_ffhq_karras2020.png}}
@karras2020training


# Generative Models, Density Modeling

- We model the underlying data distribution of the data $p(x)$ with a surrogate distribution $q(x)$, i.e.
$q(x) \thickapprox p(x)$ for some similarity between probability distributions
- In generative modeling, $x$ is in general a high-dimensional vector (e.g. image, sound, text)
- A common similarity measure used is the Kullback–Leibler ($\text{KL}$) divergence:
$$ D_{\text{KL}} (p \parallel q) = \int p(x)\log\frac{p(x)}{q(x)}dx $$

# Generative Models, KL Divergence

- Connection to information theory and compression:\
$D_{\text{KL}} (p \parallel q)  = (-\int p(x)\log q(x)dx) - (-\int p(x)\log p(x)dx) = H(p, q) - H(p)$
- Connection to maximum likelihood: For a parametrized model $q_{\Theta}$, maximization of the likelihood
$$ \text{max}_{\Theta}[E_{x \sim p} \log q_{\Theta}(x)] \thickapprox \text{max}_{\Theta}[\frac{1}{|D|} \sum_{x_i \in D} \log q_{\Theta}(x_i)] $$
is equivalent to minimizing the $\text{KL}$ divergence between $q_{\Theta}$ and $p$

# Taxonomy of Generative Models

\center{\includegraphics[width=0.8\textwidth]{images/taxonomy.png}}

# Applications: Image Recovery

- We can upscale images, denoise them, fill/recover missing parts (inpainting)
- In this case, we model $p(x|\tilde{x})$ where $\tilde{x}$ is $x$ with some information destroyed ,e.g. by blurring
or noise introduction, and we would like to recover that information
- Examples: TecoGAN [@Chu_2020], DeblurGAN [@kupyn2018deblurgan], Palette [@saharia2021palette]

\center{\includegraphics[width=0.35\textwidth]{images/image_recovery.png}}

# Applications: Conditional Image Generation

- We can generate images based on labels, or feature vectors, or natural language
- In this case, we model $p(x|y)$, where $x$ is the image and $y$ is the label, represented as a feature vector, a category, or a sequence of tokens ($y=y_1,...,y_m$, where $m$ is the number of tokens)

\center{\includegraphics[width=0.8\textwidth]{images/biggan.png}}
[@brock2019large]

# Applications: Conditional Image Generation

- What would a "penguin made of apples" look like?

\center{\includegraphics[width=0.7\textwidth]{images/DALL_e.png}}
OpenAI's DALL-E <https://openai.com/blog/dall-e/>, also see GLIDE [@nichol2021glide]

# Applications: Image-to-Image Models

- We can learn a mapping from images to images or from videos to videos, either with paired samples or unpaired ones.
Examples: Pix2Pix [@isola2018imagetoimage], CycleGAN [@zhu2020unpaired], Pix2PixHD [@wang2018highresolution], SPADE [@park2019semantic], StarGANv2 [@choi2020stargan], Palette [@saharia2021palette]


\center{\includegraphics[width=0.7\textwidth]{images/image_to_image.png}}

# Applications: Speech Synthesis

- WaveNet

\center{\includegraphics[width=0.8\textwidth]{images/speech.png}}

[@oord2016wavenet]

- Tacotron2

\center{\includegraphics[width=0.3\textwidth]{images/tacotron2.png}}

[@shen2018natural]

# Applications: Music generation

- Jukebox (<https://openai.com/blog/jukebox/>)

\center{\includegraphics[width=0.9\textwidth]{images/jukebox.png}}

[@dhariwal2020jukebox]

# Applications: Text generation

- GPT-3 [@brown2020language]

\center{\includegraphics[width=0.9\textwidth]{images/gpt3_examples.png}}

Source: <https://bit.ly/3tqM8Sr>, <https://bit.ly/39KHNSe>, <https://bit.ly/3tvipHR>

# Applications: Reinforcement Learning

- We can simulate possible futures given the past. Application for reinforcement learning: learning world models and planning

\center{\includegraphics[height=6cm]{images/reinforcement_learning.png}}

[@hafner2020dream], [@kim2020learning]

# Generative Adversarial Networks

- We have samples from a probability distribution $p$, and we would like
to learn a generator model that generates samples from $x \sim p(x)$
- Discriminator is trained to distinguish between real and generated samples
- Generator is trained to fool the discriminator
<!--- Generator takes as input a randomly sampled vector $\boldsymbol{z}$ and outputs a sample $\boldsymbol{x}$-->
<!--- Discriminator receives either real or genereted images as input-->
<!--- Discriminator loss: predict that real data is real and fake data is fake-->
<!--- Generator loss tries to fool the discriminator: make the discriminator predict that fake data is real-->

\center{\includegraphics[width=0.7\textwidth]{images/gan.png}}
<!--Source: https://developers.google.com/machine-learning/gan/gan_structure-->

# Generative Adversarial Networks – Formulation

$$ \min _{G} \max _{D} \left[\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]\right] $$

- $D(x)$: probability that an image is real
- $D(G(z))$: probabilty that a generated image is real, where $z \sim \mathbb{N}(0, I)$
- $\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]$: discriminator on real data
- $\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]$: discriminator on generated data

# Generative Adversarial Networks – Ideal Case

- Generative Adversarial Networks [@goodfellow2014generative] shows that for a fixed generator $G$, the optimal
discriminator is: $D^{*}_{G}(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}} + p_{g}(x)}$
- They also show that the global optimal solution of the problem minimizes: $-\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g )$, where the $\text{JSD}$ is a measure between probability distributions
- Since the $\text{JSD}$ is non-negative, the globally optimal solution for the generator is the data distribution $p_g = p_{\text{data}}$

# Generative Adversarial Networks – Training

- In practice, we alternate between updating the generator and updating the discriminator

\center{\includegraphics[height=5cm]{images/gan_training.png}}

- Note that in practice, for the generator they maximize $\log (D(G(\boldsymbol{z})))$ instead of minimizing $\log (1-D(G(\boldsymbol{z})))$,
because it provides better gradients

# Generative Adversarial Networks – Issues

- Non-convergence
- Mode collapse
- Vanishing gradient

\center{\includegraphics[width=0.9\textwidth]{images/collapse.png}}

(Illustration of mode collapse)

# The DCGAN Architecture

- Removed fully connected layers: fully convolutional architecture for generator and discriminator
- Uses batch normalization to stabilize training
- One of the first architectures that worked well in practice on several datasets
\center{\includegraphics[width=0.9\textwidth]{images/dcgan.png}}

# The DCGAN Architecture – Vector Arithmetics

- Interpretable directions in the latent space

\center{\includegraphics[width=0.9\textwidth]{images/dcgan_vector_arithmetics.png}}

# The DCGAN Architecture – Interpolation in Latent Space

- Smooth interpolation between generated images using the latent space

\center{\includegraphics[width=0.9\textwidth]{images/dcgan_interp1.png}}

\center{\includegraphics[width=0.9\textwidth]{images/dcgan_interp2.png}}

# Evaluation Metrics

- In general, it's still an open question how to evaluate a generative model; in a lot of cases human visualization is still needed
- In the ideal case, metrics should be task-specific [@theis2016note] and evaluate your generative model depending on how you will use it
- Most common metrics used: Fréchet Inception Distance (FID) [@heusel2018gans], Inception Score [@salimans2016improved], precision and recall [@kynkäänniemi2019improved]

# Summary

- Impressive progress during last years
- Model sizes and data are getting bigger
- Lot of different applications: image generation, text generation, speech synthesis, and more

# References {.allowframebreaks}
