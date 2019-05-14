# Pong from Pixels with tf.keras

This is an implementation of the famous [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) code with TensorFlow 2.0 and the Keras API (i.e. tf.keras).

The main reason for this is to show how easy it to create a neural network that can play a game. The boiler plate code around the Gym environment is more or less the same as the original. However, we process all 10 episodes in one batch instead of each episode separately. In the original code the policy gradients are computed after every episode ended and stored until 10 episodes have been processed. Then one single RMSProp step is performed. In this implementation a whole batch of 10 episodes is given to the model to train on.

## Installation

All development was done with Ubuntu, since the Gym environment doesn't really work with Windows.
The following applications/packages were used:
1) [Miniconda](https://conda.io/en/latest/miniconda.html)
2) [TensorFlow 2.0 (alpha)](https://www.tensorflow.org/install)
3) [OpenAI Gym](https://gym.openai.com/docs/#installation)


## Results

This implementation stores all episode rewards in a log file. The following plot shows all the rewards after one night of training (~12 hours, 6500+ episodes) and the moving average (with windows size 250) over all these rewards.

![Rewards after 1 night of training](../assets/rewards-1-night.png?raw=true)
