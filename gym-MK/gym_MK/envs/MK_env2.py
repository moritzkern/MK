#loading requiered packages
import copy
import logging
logger = logging.getLogger(__name__)
import gym
from gym.utils import seeding
import numpy as np
import pandas as pd 
from helper_stats import state, min_prod, demand_dist, demand_dist_parameter

# import config csv 
df = pd.read_csv("/Users/andreferdinand/Desktop/MOPT/MK/gym-MK/gym_MK/envs/Products.csv")

# import startespace as dictionary
start_space = dict(["STORAGE_"+i, int(df["STORAGE"][df["PRODUCT"]==i])] for i in df["PRODUCT"])
start_space.update(dict(["DEMAND_"+j, 0] for j in df["PRODUCT"][df["ENDPRODUCT"]]))

# dictionary with dependencies between products
dependencies_dict = dict()
gen = (j for j in df.columns if j.startswith("PROCESS"))
for j in gen:
    dependencies_dict.update(pd.Series(df[df[j].notnull()].PRODUCT.values,index=getattr(df[df[j].notnull()],j)).to_dict())


# dictionary with different demand distributions 
_DIST = {
    "NORMAL": np.random.normal,
    "UNIFORM": np.random.uniform
}


class MKEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        
        #TODO:

        # List of all Endproducts
        self.endproducts = df["PRODUCT"][df["ENDPRODUCT"]].to_list()

        # Setting demand distribution of Endproducts
        self.demand_dist_end = demand_dist()
        self.demand_dist_end_para = demand_dist_parameter()
        
        # Names of state space variables
        self.names = ["STORAGE_"+i for i in df["PRODUCT"]] + ["DEMAND_"+j for j in df["PRODUCT"][df["ENDPRODUCT"]]]

        # dictionary with assignment of position from products and demands   
        self.observation_assignment = dict(enumerate(self.names))
        self.observation_assignment_rev = {y:x for x,y in self.observation_assignment.items()}
        # number of possible (different) states for each object
        self.observation_number_of_possibilities = 200

        # Initialize the game
        self._seed()
        self.game_state = self.reset()
        self.game_state_length = len(self.game_state)
        
        # Construction of observable space

        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.observation_number_of_possibilities) for _ in range(self.game_state_length)]))

        # Minimum quantity to produce 
        self.min_product = min_prod()

        # an action consists of setting the production quantities of all individual products --> here 4 products 
        # dictionary with assignment of position from products and demands 
        self.action_assignment = dict(enumerate(self.endproducts))
        self.action_assignment_rev = {y:x for x,y in self.action_assignment.items()}

        # number of possible (different) actions for each endproduct
        self.action_number_of_possibilities = 5
        self.action_length = len(self.endproducts)

        # Construction of action space
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.action_number_of_possibilities) for _ in range(self.action_length)]))
        
        # step in the game

        self.count = 0

        # Penalizing stock quantities and not meeting demand 
        self.KAPPA = 1000
        self.LAMBDA = 5

        # Determination of the maximum number of steps
        self.MAX_STEPS = 100

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.done = False
        self.info = {}
        self.reward = 0
        return tuple(start_space[self.observation_assignment[i]] for i in range(len(start_space)))

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
            self.game_state = self._get_demand()
            self.game_state = self._get_prod(action)
            try:
                assert self.observation_space.contains(self.state)
            except AssertionError:
                print("INVALID STATE", self.state)
            self.reward = self._get_reward()
            self.game_state = self._update_storage()
            self.info["action"] = "|".join([i+ ": " + str(self.game_state[self.observation_assignment_rev[i]]) for i in self.names])
            self.info["action"] = "|".join([i+ ": " + str(action[self.action_assignment_rev[i]]) for i in self.endproducts])
        return [self.game_state, self.reward, self.done, self.info]

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
        print("state:")
        print("|".join([i+ ": " + str(self.game_state[self.observation_assignment_rev[i]]) for i in self.names]))
 

    def _get_demand(self):
        """
        Simulate the different demands 
        """
        liste = list(self.game_state)
        for i in self.endproducts:
            liste[self.observation_assignment_rev["DEMAND_"+i]] = _DIST[getattr(self.demand_dist_end,"DDIST_"+i)](*getattr(self.demand_dist_end_para,"DDISTPARA_"+i))
        # TODO: -only positiv demand is possible therefore clip the vector 
        #       -only integer demand is possible
        return tuple(liste)

    def _get_reward(self,action):
        """ 
        Reward for producing 
        """
        #not meeting the demand is evaluated very negatively
        if not all([self.game_state[self.observation_assignment_rev["STORAGE_"+i]]+action[self.action_assignment_rev[i]]-self.game_state[self.observation_assignment_rev["DEMAND_"+i]]>=0 for i in self.endproducts]):
            return -self.KAPPA
        #stock quantities are also not that nice
        else:
            return -self.LAMBDA * (sum([self.game_state[self.observation_assignment_rev["STORAGE_"+i]] for i in df["PRODUCT"]])+sum(action))

    def _get_prod(self,action):
        """
        TODO!!!!
        Test function
        """
        dependencies_dict_copy = copy.deepcopy(dependencies_dict)
        liste = list(self.game_state)
        action_dict = dict([[j,0] for j in df["PRODUCT"][~df["ENDPRODUCT"]]])
        for i in self.endproducts:
            liste[self.observation_assignment_rev["STORAGE_"+i]] += action[self.action_assignment_rev[i]]
            action_dict[dependencies_dict[i]] += action[self.action_assignment_rev[i]]
            dependencies_dict_copy.pop(i)
        notfinished_prod = [j for j in df["PRODUCT"][~df["ENDPRODUCT"]]]
        print(action_dict)
        while notfinished_prod:
            while list(set(notfinished_prod)-set(list(dependencies_dict_copy.values()))):
                for j in list(set(notfinished_prod)-set(list(dependencies_dict_copy.values()))):
                    print(j)
                    if liste[self.observation_assignment_rev["STORAGE_"+j]]>=action_dict[j]:
                        liste[self.observation_assignment_rev["STORAGE_"+j]] -= action_dict[j]
                    else:
                        try:
                            action_dict[dependencies_dict_copy[j]] += max(action_dict[j]-liste[self.observation_assignment_rev["STORAGE_"+j]], getattr(self.min_product,"MIN_"+j))
                            liste[self.observation_assignment_rev["STORAGE_"+j]] += -action_dict[j] + action_dict[dependencies_dict_copy[j]]
                        except:
                            liste[self.observation_assignment_rev["STORAGE_"+j]] += -action_dict[j] + max(action_dict[j]-liste[self.observation_assignment_rev["STORAGE_"+j]],getattr(self.min_product,"MIN_"+j))
                    notfinished_prod.remove(j)
                    try: 
                        dependencies_dict_copy.pop(j)
                    except:
                        pass
        return tuple(liste)
    
    def _update_storage(self):
        """substract demand from storage"""
        liste = list(self.game_state)
        for i in self.endproducts:
            liste[self.observation_assignment_rev["STORAGE_"+i]] = max(liste[self.observation_assignment_rev["STORAGE_"+i]]-liste[self.observation_assignment_rev["DEMAND_"+i]],0) 
        return tuple(liste)