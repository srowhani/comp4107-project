
## Description

Our project compares several variants of convolutional neural networks to
evaluate classification accuracy relative to configuration. We look at several
architectures, employing different tactics to achieve better results.

## Group Member
Seena Rowhani - 100 945 353
Chris Ermel - 100 934 583

## Dependencies
```
import warnings
import _pickle as pickle
from __future__ import division
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from math import sqrt
import matplotlib.pyplot as plt
from scipy.misc import toimage
```

Note - `tensorflow-gpu` was used.

## Testing

Each network can be tested through `jupyter`, or by running the python file.
Classification accuracy can be viewed on tensorboard post-completion - and Estimator
model checkpoints are computed along the way.
