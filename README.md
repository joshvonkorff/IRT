# Deep Knowledge Tracing and Item Response Theory

Deep Knowledge Tracing involves applying Recurrent Neural Networks to a set of student responses on a set of problems, attempting to predict the responses in advance using the previous responses:

https://proceedings.neurips.cc/paper/2015/file/bac9162b47c56fc8a4d2a519803d51b3-Paper.pdf

The RNN encodes a given problem as a one-hot encoding, with another column for each response.  Thus, if a given student answers problem 1 correctly and problem 2 incorrectly, the input to the RNN would be:

[1, 1, 0, 0]
[0, 0, 1, 0]

The first column is problem 1, the second is the answer to problem 1 (correct or incorrect), and so on.

In this project, I have used an Item Response Theory model (with student abilities and problem difficulties) to create student responses.  The student ability in this case has only a single latent variable, so that I can test the result against the loss of regular Item Response Theory.

I obtained the idea for the code from:
https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
And from:
Deep Learning with PyTorch, by Eli Stevens (Author), Luca Antiga (Author), Thomas Viehmann (Author)
Which is publicly available here: https://github.com/deep-learning-with-pytorch/dlwpt-code

However, neither of these sources uses Deep Knowledge Tracing, so only the basic idea of this code comes from these sources.

In the notebook file, you can see that over the courses of several epochs, the MSE validation loss decreases significantly.  I have also included an item response theory MSE loss calculation, using the py-irt code.  At the time I'm writing this, the DKT loss is better than the IRT loss.
