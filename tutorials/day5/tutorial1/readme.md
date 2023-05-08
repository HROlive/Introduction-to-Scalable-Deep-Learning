Day 5 Tutorial 1 
===================

In this tutorial, you will learn how to:

- Train a [Deep Convolutional Generative Adversarial Network](https://arxiv.org/pdf/1511.06434.pdf) (DCGAN), which consists of a convolutional network-based **generator** and **discriminator**.
- Distribute the training using Horovod's [DistributedGradientTape](https://horovod.readthedocs.io/en/stable/api.html#horovod.tensorflow.DistributedGradientTape) wrapper for the [gradient tapes](https://www.tensorflow.org/api_docs/python/tf/GradientTape?version=nightly)
