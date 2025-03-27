# Interview test
You're given a simple autograd library running CPU, written in numpy, with
pytorch as a reference.
Repository: https://github.com/loctxmoreh/simple-grad

Your task is to implement new features from the below feature list. 

## Due date 
3 days since the day you receive this assignment

## Feature list
1. Implement elementwise subtraction & division & negation
2. Implement one activation function of your choice (hint: softmax)
3. Implement conv2d ops and its corresponding module
    * For simplicity, assume:
        * input layout: channel first (BCHW)
        * H == W (square image)
        * no padding
        * stride=1, dilation=1
        * bias=True
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

Note: 
* Each feature should have corresponding unittest, showing intended usage &
  behavior 
* You can implement features in any order
* Implementing all of them is **not** required
    * You are evaluated by quality first, quantity second

## Evaluation method
1. Clone the repository to your local machine, and then push it to a
   **private** repository of yours
2. Invite @loctxmoreh, @long10024070 and @mrshaw01 as collaborator to your
   private repository
3. For each feature, develop on a separate branch than main/master. Once
   finished, open a PR to your own repo and request three collaborators as
   reviewer.
    * PR opened later that the due date will not be reviewed.
4. We'll have discussion back and forth
5. Merging to original repository is not required
    * But you can do whatever you want with your own repo. Code is MIT-licensed 

