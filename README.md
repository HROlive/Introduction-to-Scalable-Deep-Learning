Introduction to Scalable Deep Learning
===========

## Description

In this course, we will cover machine learning and deep learning and how to achieve scaling to high performance computing systems. The course aims at covering all levels, from fundamental software design to specific compute environments and toolkits. We want to enable the participants to unlock the resource of machines like the JUWELS booster for their machine learning workflows. Different from previous years we assume that the participants have a background from a university level introductory course to machine learning. Suggested options for self-teaching are given below.

We will start the course with a presentation of high performance computing system architectures and the design paradigms for HPC software. In the tutorial, we familiarize the users with the environment. Furthermore, we give a recap of important machine learning concepts and algorithms and the participants will train and test a reference model. Afterwards, we introduce how deep learning algorithms can be parallelized for supercomputer usage with Horovod. Furthermore, we discuss best practicies and pitfalls in adopting deep learning algorithms on supercomputers and learn to test their function and performance. Finally we apply the gained expertise to large scale unsupervised learning, with a particular focus on Generative Adversarial Networks (GANs).

## Basic Info

This repo contains the material for the course Scalable Deep Learning.

It is a five half-day course with two lectures and two tutorials each.
Lectures and tutorials are located in the corresponding folders [lectures](lectures), [tutorials](tutorials) .

All Tutorials are set up in the form of jupyter notebooks. Only some tutorials should be executed
in notebooks as well. Most tutorials rely on adapting code in standalone scripts.
Therefore, we strongly suggest the workflow of cloning the repo on our supercomputer
and executing the scripts there.

[Jupyter-JSC](https://jupyter-jsc.fz-juelich.de/) can be used to view and execute the tutorial code.

## Syllabus Outline

### Day 1


#### Lecture 1
- [Intro](https://mldl_fzj.pages.jsc.fz-juelich.de/juhaicu/jsc_public/sharedspace/teaching/intro_scalable_deep_learning/course-material-may-2023/#/title-slide)
- [Getting started on a supercomputer](https://mldl_fzj.pages.jsc.fz-juelich.de/juhaicu/jsc_public/sharedspace/teaching/intro_scalable_deep_learning/course-material-may-2023/01-access-machines.html#/title-slide)
  - Content: Stefan Kesselheim

#### Tutorial 1
- [First Steps on the Supercomputer](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day1/tutorial1)
  - Content: Stefan Kesselheim, Jan Ebert
  - Content supervisor: Stefan Kesselheim

#### Lecture 2
- [Supercomputer architecture and MPI Primer](https://mldl_fzj.pages.jsc.fz-juelich.de/juhaicu/jsc_public/sharedspace/teaching/intro_scalable_deep_learning/course-material-may-2023/02-mpi.html#/title-slide)
  - Content: Stefan Kesselheim

#### Tutorial 2
- [Hello MPI World](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day1/tutorial2)
  - Content: Jan Ebert, Stefan Kesselheim
  - Content supervisor: Stefan Kesselheim


### Day 2

#### Lecture 1
- [Intro Large-Scale Deep Learning - Motivation, Deep Learning Basics Recap](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day2/Lecture1/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

#### Tutorial 1
- [Deep Learning Basics Recap](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day2/tutorial1)
  - Content: Roshni Kamath, Jenia Jitsev
  - Content supervisor: Jenia Jitsev

#### Lecture 2
- [Distributed Training and Data Parallelism with Horovod](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day2/Lecture2/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

#### Tutorial 2
- [Dataset API and Horovod Data Parallel Training Basics](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day2/tutorial2)
  - Content: Jan Ebert, Stefan Kesselheim, Jenia Jitsev
  - Content supervisor: Jenia Jitsev

### Day 3

#### Lecture 1
- [Scaling Laws and Training with Large Data](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day3/Lecture1/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

#### Tutorial 1:
- [Distributed Training - Throughput and Scaling](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day3/tutorial1)
  - Content: Mehdi Cherti, Jenia Jitsev
  - Content supervisor: Jenia Jitsev

#### Lecture 2
- [Is My Code Fast? Performance Analysis](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day3/Lecture2/Slides)
  - Content: Stefan Kesselheim

#### Tutorial 2:
- [Data Pipelines and Performance Analysis](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day3/tutorial2)
  - Content: Jan Ebert, Stefan Kesselheim
  - Content supervisor: Stefan Kesselheim



### Day 4

#### Lecture 1
- [Combating Accuracy Loss in Distributed Training](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day4/Lecture1/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

#### Tutorial 1:
- [Combating Accuracy Loss in Distributed Training](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day4/tutorial)
  - Content: Mehdi Cherti
  - Content supervisor: Jenia Jitsev

#### Lecture 2
- [Advanced Distributed Training and Large-Scale Deep Learning Outlook](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day4/Lecture2/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev


### Day 5

#### Lecture 1
- [Generative Adversarial Networks (GANs) basics](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day5/Lecture1.pdf)
  - Content: Mehdi Cherti

#### Tutorial 1
- [Basic GAN distributed training using Horovod](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day5/tutorial1)
  - Content: Mehdi Cherti
  - Content supervisor: Mehdi Cherti

### Lecture 2
- [Advanced Generative Adversarial Networks](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/lectures/Day5/Lecture2.pdf)
  - Content: Mehdi Cherti

### Tutorial 2
- [Advanced GAN distributed training using Horovod](https://github.com/HROlive/Introduction-to-Scalable-Deep-Learning/tree/main/tutorials/day5/tutorial2)
  - Content: Mehdi Cherti
  - Content supervisor: Mehdi Cherti
