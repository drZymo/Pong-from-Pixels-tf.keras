import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
import gym
import numpy as np
from pathlib import Path


WeightsFile = Path('pong-weights.h5')
RewardLogFile = Path('pong-reward.log')
EpisodeSaveInterval = 20
NrOfEpisodesPerEpoch = 10
Gamma = 0.99

        
def BuildModels():
    observation = Input((6400,), name='observation')
    x = Dense(200, activation='relu', name='dense')(observation)
    action = Dense(1, activation='sigmoid', name='action')(x)
    model_play = Model(observation, action)

    reward = Input((1,), name='reward')
    
    def policy_gradient_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon()) # prevent log of 0
        bce = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred) # binary cross entropy
        return -K.mean(bce * reward)
    
    model_train = Model([observation, reward], action)
    model_train.compile(loss=policy_gradient_loss, optimizer='adam')

    if WeightsFile.exists():
        print("Loading previous weights")
        model_train.load_weights(str(WeightsFile), by_name=True)

    return model_play, model_train


rewardLogFile = open(RewardLogFile, 'a+')

def LogRewards(reward):
    rewardLogFile.write('{0:f}\n'.format(reward))
    rewardLogFile.flush()


# Start environment
env = gym.make("Pong-v0")

# Create model
model_play, model_train = BuildModels()
model_play.summary()


def ProcessObservation(observation):
    ob = observation[35:195] # crop
    ob = ob[::2,::2,0] # downsample by factor of 2
    ob[ob == 144] = 0 # erase background (background type 1)
    ob[ob == 109] = 0 # erase background (background type 2)
    ob[ob != 0] = 1 # everything else (paddles, ball) just set to 1
    ob = ob.astype(np.float16) # uint8 to float
    ob = ob.reshape(-1) # flatten
    return ob


def DiscountRewards(rewards, discountFactor):
    discountedRewards = np.zeros((len(rewards)))
    currentReward = 0
    for i in reversed(range(len(rewards))):
        if rewards[i] != 0: currentReward = 0  # reset the sum, since this was a game boundary (pong specific!)
        currentReward = rewards[i] + discountFactor * currentReward
        discountedRewards[i] = currentReward
    return discountedRewards


def RunEpisode():
    observations, actions, rewards = [], [], []

    # Initialize environment
    observation_new = env.reset()
    observation_new = ProcessObservation(observation_new)
    observation_old = observation_new

    # Start playing until done
    done = False
    while not done:
        # Compute difference in observations
        observation = observation_new - observation_old
        observations.append(observation)

        # Predict and choose an action based on current observation
        actionProbability = model_play.predict(np.expand_dims(observation, axis=0))[0]
        action = 1 if np.random.uniform() < actionProbability else 0  # 1 = up, 0 = down
        actions.append(action)

        # Take the action and get reward and new observation
        observation_old = observation_new
        observation_new, reward, done, info = env.step(2 if action == 1 else 3) # 2 is up, 3 is down
        observation_new = ProcessObservation(observation_new)
        rewards.append(reward)

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)

    return observations, actions, rewards, np.sum(rewards)


# Train
episode = 0
while True:
    batchObservations, batchActions, batchRewards = [], [], []

    for _ in range(NrOfEpisodesPerEpoch):
        print('ep #{0}: '.format(episode), end='')
        
        # Run one episode
        episodeObservations, episodeActions, episodeRewards, episodeTotalReward = RunEpisode()

        # Discount rewards over time and normalize
        episodeRewards = DiscountRewards(episodeRewards, Gamma)
        episodeRewards -= np.mean(episodeRewards)
        episodeRewards /= np.std(episodeRewards)

        # Add to batch
        batchObservations.append(episodeObservations)
        batchActions.append(episodeActions)
        batchRewards.append(episodeRewards)

        LogRewards(episodeTotalReward)
        print('length {}, reward {}'.format(len(episodeActions), episodeTotalReward))

        episode += 1

    batchObservations = np.concatenate(batchObservations)
    batchActions = np.expand_dims(np.concatenate(batchActions), axis=1)
    batchRewards = np.expand_dims(np.concatenate(batchRewards), axis=1)

    # Train on batch
    model_train.train_on_batch([batchObservations, batchRewards], batchActions)
    if episode % EpisodeSaveInterval == 0:
        model_train.save_weights(str(WeightsFile))
