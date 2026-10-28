import random

class replay_buffer():
  """Stores enviroment transitions"""

  def __init__(self):
    self.replay_buffer=[]

  def add(self, state, action, reward,new_state, terminated, truncated):
    """Store a new transition in the replay buffer"""
    self.replay_buffer.append((state, action, reward, new_state, float(terminated), float(truncated)))

  def get_random_sample(self, n):
    """Returns a random batch of n transitions for training"""
    random_sample_list=random.sample(self.replay_buffer,n)
    return random_sample_list

  def get_buffer_size(self):
    """Returns the number of transitions in the buffer"""
    return len(self.replay_buffer)

  