# Background separation in images

In this group project we explore AI image processing techniques for **separating the subject from the background in images**, also known as **alpha matting**. The goal is to build on existing models and methods to get very **accurate** results.

We use the **trimap** approach, in which the alpha channel is generated in two separate steps.

1. First, a ***trimap*** is generated. This is a 3-channel image that segmentates the original image into
    * the background (black);
    * the foreground (white);
    * the edge zone where the next model should do its thing (gray).
2. From this trimap, another model generates the final alpha channel.

This project is a collaboration between [Bram](https://github.com/), [Martin](https://github.com/), [Philippe](https://github.com/), and is part of a training in collaboration with [Faktion](https://www.faktion.be). It was completed in two weeks.

## Installation & usage

## Technical explanation

### About the training data

We use de [DUTS](http://saliencydetection.net/duts/) dataset for training our models. It consists of  

We created two wrapper classes in _[duts.py](duts.py)_ for easy working with the structure of this data.

### Trimap generation

### Alphamatting
