import logging
logger = logging.getLogger(__name__)
import gym
from gym.utils import seeding
import numpy as np
import pandas as pd 
from state import State
df = pd.read_csv("/Users/andreferdinand/Desktop/MOPT/MK/gym-MK/gym_MK/envs/Products.csv")

state = State()


class MKEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        #how much must be produced at least of all products
        self.state =State()
        #intermediate products
        self.MIN_DOUGH = 100
        self.MIN_BUNS = 70

        # end products
        self.MIN_BREAD = 70
        self.MIN_BUNS1 = 10
        self.MIN_BUNS2 = 10
        self.MIN_BUNS3 = 10 

        #Initial situation

        #Initial stocks
        self.WH_DOUGH = 0
        self.WH_BUNS = 0
        self.WH_BREAD = 0
        self.WH_BUNS1 = 0
        self.WH_BUNS2 = 0
        self.WH_BUNS3 = 0
        #Initial demand
        self.D_BUNS1 = 0
        self.D_BUNS2 = 0
        self.D_BUNS3 = 0
        self.D_BREAD = 0

        # an action consists of setting the production quantities of all individual products --> here 6 products 
        self.action_space = gym.spaces.Discrete(4)
        
        # all stocks (intermediate products + end products) and demands (all products) of the last period are observed  
        self.observation_space = gym.spaces.Discrete(10)

        # Initialization of the game
        self._seed()
        self._reset()

        # step in the game

        self.count = 0

        # Penalizing stock quantities and not meeting demand 
        self.KAPPA = 1000
        self.LAMBDA = 5

        # Determination of the maximum number of steps
        self.MAX_STEPS = 100

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.state = np.array([self.WH_DOUGH,self.WH_BUNS,self.WH_BREAD,self.WH_BUNS1,self.WH_BUNS2,self.WH_BUNS3, self.D_BREAD,self.D_BUNS1,self.D_BUNS2,self.D_BUNS3])
        self.done = False
        self.info = {}
        self.reward = 0
        return self.state

    def _step(self, action):
        """

        Parameters
        ----------
        action :

        Returnss
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.done:
            print("EPISODE DONE!")
        elif self.count == self.MAX_STEPS:
            self.done =True
        else: 
            assert self.action_space.contains(action)
            self.count +=1
            self.state[6:] = self._get_demand()
            self.state[:5] = self._get_prod(action)
            try:
                assert self.observation_space.contains(self.state)
            except AssertionError:
                print("INVALID STATE", self.state)
            self.reward = self._get_reward()
            self.info["action"] = "BREAD {:2d}, BUNS1 {:2d}, BUNS2 {:2d}, BUNS3 {:2d}".format(self.action[0],self.action[1],self.action[2],self.action[3])
       
        return [self.state, self.reward, self.done, self.info]

    def _seed(self,seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        Args:
            mode (str): the mode to render with
        """
        s = "state: {:2d}  reward: {:2d}  info: {}"
        print(s.format(self.state, self.reward, self.info))

    def _get_demand(self):
        """
        simulate the different demands 
        """
        D_bread = np.random.normal(loc=3,scale=0.5,size=1)
        D_buns1 = np.random.normal(loc=2,scale = 0.2,size=1)
        D_buns2 = np.random.normal(loc=1,scale=0.1,size=1)
        D_buns3 = np.random.normal(loc=1,scale=0.1,size=1)
        # only positiv demand is possible therefore clip the vector 
        return np.clip(np.array[D_bread,D_buns1,D_buns2,D_buns3],0,None)

    def _get_reward(self):
        """ Reward for producing """
        #not meeting the demand is evaluated very negatively
        if self.state[2:5]-self.state[5:]:
            return -self.KAPPA
        #stock qunatities are also not that nice
        else:
            return -self.LAMBDA * np.sum(self.state[:5])

    def _get_prod(self,action):
        """
        TODO!!!!
        what is the easiest way to serve the production MK help?
        """
        if 5:
            pass


        prod_bread = 0
        prod_dough = 0
        return 