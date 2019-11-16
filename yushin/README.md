# Run requirments

* Installed packages(numpy, sklearn)
* data(e.g. train-images.idx3-ubyte) should be in ./dataset directory

```
$ python mysvm_test.py
```



# Current progress

## Basic training & test data set

* Training data: train-images.idx3-ubyte
* Test data: t10k-images.idx3-ubyte

-> 83%~84%



## Basic training data + new test data

* Training data: train-images.idx3-ubyte
* Test data: mnist-new1k-images-idx3-ubyte

->~72%

## Mixed

- Training data: train-images.idx3-ubyte
- Test data: t10k-images.idx3-ubyte
- Shuffled
- PCA = 0.95
- max_iter = 1000
- RMSProp Optimizer
- 15,000 training data with CV
- {'c': 125, 'eta': 0.005, 'forget_factor': 0.5} OR {'c': 100, 'eta': 0.005, 'forget_factor': 0.8}

-> 88.6%~88.7%