## Requirements
In this branch, I did the following:

4. Implement pytorch-style SGD optimizer
    * Only learning rate, no momentum
5. Implement single-step training for a simple two-layer network
    * Architecture: linear - relu - linear - (your activation of choice)
    * Problem: arbitrary (preferably: classification)
    * Hidden shapes: arbitrary
    * Dataset: arbitrary
    * Shows that loss decreases after one forward-backward pass
    * Implement a training loop and show network convergence
        * In a standalone python script, using `simplegrad` as a dependency

-----
As the only way to tell whether an optimizer is working properly is to applying it to a training loop. Therefore I developed two requirements (4, 5) in to a single branch.

## Features implemented:
0. (optional) I implemented some more ops for convenience, including `reshape`, `broadcast_to`, `summation`.

1. Schocastic Gradient Descent without momentum.
    * To test my understanding, I also add L2 regularization via weight_decay.

2. (optional) A functional Adam optimizer. As my SGD optimizer did not converges really well, I was motivated to try implement Adam.

3. A simple neural network.
   * Architecture: linear - relu - linear - CrossEntropy loss
   * Problem: Teach the model to learn how to convert decimal numbers to binary numbers
   * Dataset: Preparing data is easy as I can generate as many numbers as I want. Specifically, I get all pairs of (decimal, binary) with decimal < 4096. I do train_test_split with ratio 9:1.
   * Training loop: I purely use `simplegrad` for training. I did use SGD but it converged really slow and seemed to be stuck at local minima, therefore I tried Adam. After training for 1000 epochs with Adam optimizer, the model converges, reach near perfect accuracy. You can peak into the training loop as follows:

```
Epoch 0: Training loss: 0.6545, Number accuracy: 0.24%, Digit accuracy: 72.06
Epoch 50: Training loss: 0.2509, Number accuracy: 3.41%, Digit accuracy: 82.35
Epoch 100: Training loss: 0.1960, Number accuracy: 10.98%, Digit accuracy: 86.53
...
Epoch 900: Training loss: 0.0020, Number accuracy: 95.61%, Digit accuracy: 99.66
Epoch 950: Training loss: 0.0015, Number accuracy: 96.59%, Digit accuracy: 99.74

Illustration:
Example 1:
  Decimal number  : [2, 7, 8, 1]
  Target binary   : [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
  Predicted binary: [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]

Example 2:
  Decimal number  : [2, 6, 2, 8]
  Target binary   : [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
  Predicted binary: [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]

```

## Testing

I custom unit test functions instead of relying on pytest for more verbose logging, which is easier for debugging. To test the implemented features:
```
python my_tests/test_optim.py
python my_tests/test_ops.py
python scripts/train.py
```
