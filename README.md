Introduction to Scalable Deep Learning
===========

# Basic Info

This repo contains the material for the course Scalable Deep Learning.

It is a five half-day course with two lectures and two tutorials each.
Lectures and tutorials are located in the corresponding folders [lectures](lectures), [tutorials](tutorials) .

All Tutorials are set up in the form of jupyter notebooks. Only some tutorials should be executed
in notebooks as well. Most tutorials rely on adapting code in standalone scripts.
Therefore, we strongly suggest the workflow of cloning the repo on our supercomputer
and executing the scripts there.

[Jupyter-JSC](jupyter-jsc.fz-juelich.de/) can be used to view and execute the tutorial code.

# Syllabus Outline

## Day 1


### Lecture 1
- [Intro](lectures/Day1/Intro/Slides/slides.pdf)
- [Getting started on a supercomputer](lectures/Day1/Lecture1/Slides/slides.pdf)
  - Content: Stefan Kesselheim

### Tutorial 1
- [First Steps on the Supercomputer](tutorials/day1/tutorial1)
  - Content: Stefan Kesselheim, Jan Ebert
  - Content supervisor: Stefan Kesselheim

### Lecture 2
- [Supercomputer architecture and MPI Primer](lectures/Day1/Lecture2/Slides/slides.pdf)
  - Content: Stefan Kesselheim

### Tutorial 2
- [Hello MPI World](tutorials/day1/tutorial2)
  - Content: Jan Ebert, Stefan Kesselheim
  - Content supervisor: Stefan Kesselheim


## Day 2

### Lecture 1
- [Intro Large-Scale Deep Learning - Motivation, Deep Learning Basics Recap](lectures/Day2/Lecture1/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

### Tutorial 1
- [Deep Learning Basics Recap](tutorials/day2/tutorial1)
  - Content: Roshni Kamath, Jenia Jitsev
  - Content supervisor: Jenia Jitsev

### Lecture 2
- [Distributed Training and Data Parallelism with Horovod](lectures/Day2/Lecture2/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

### Tutorial 2
- [Dataset API and Horovod Data Parallel Training Basics](tutorials/day2/tutorial2)
  - Content: Jan Ebert, Stefan Kesselheim, Jenia Jitsev
  - Content supervisor: Jenia Jitsev

## Day 3

### Lecture 1
- [Scaling Laws and Training with Large Data](lectures/Day3/Lecture1/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

### Tutorial 1:
- [Distributed Training - Throughput and Scaling](tutorials/day3/tutorial1)
  - Content: Mehdi Cherti, Jenia Jitsev
  - Content supervisor: Jenia Jitsev

### Lecture 2
- [Is My Code Fast? Performance Analysis](lectures/Day3/Lecture2/Slides/Lecture_Slides.pdf)
  - Content: Stefan Kesselheim

### Tutorial 2:
- [Data Pipelines and Performance Analysis](tutorials/day3/tutorial2)
  - Content: Jan Ebert, Stefan Kesselheim
  - Content supervisor: Stefan Kesselheim



## Day 4

### Lecture 1
- [Combating Accuracy Loss in Distributed Training](lectures/Day4/Lecture1/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev

### Tutorial 1:
- [Combating Accuracy Loss in Distributed Training](tutorials/day4/tutorial)
  - Content: Mehdi Cherti
  - Content supervisor: Jenia Jitsev

### Lecture 2
- [Advanced Distributed Training and Large-Scale Deep Learning Outlook](lectures/Day4/Lecture2/Slides/Lecture_Slides.pdf)
  - Content: Jenia Jitsev


## Day 5

### Lecture 1
- [Generative Adversarial Networks (GANs) basics](lectures/Day5/Lecture1.pdf)
  - Content: Mehdi Cherti

### Tutorial 1
- [Basic GAN distributed training using Horovod](tutorials/day5/tutorial1)
  - Content: Mehdi Cherti
  - Content supervisor: Mehdi Cherti

### Lecture 2
- [Advanced Generative Adversarial Networks](lectures/Day5/Lecture2.pdf)
  - Content: Mehdi Cherti

### Tutorial 2
- [Advanced GAN distributed training using Horovod](tutorials/day5/tutorial2)
  - Content: Mehdi Cherti
  - Content supervisor: Mehdi Cherti
