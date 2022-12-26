from BSE import Trader_PRZI

import random
import sys
import math
from scipy.stats import cauchy

class Trader_PRJ(Trader_PRZI):

    def __init__(self, ttype, tid, balance, params, time):

        Trader_PRZI.__init__(self, ttype, tid, balance, params, time)

        self.archive = []    # stores discarded strategies
        self.display = []    # successful F strats
        self.F = params['F'] # differential evolution weight
        self.k = params['k'] # number of strategies
        self.c = params['c'] # 
        self.p = params['p']
        self.muF = 0.5 # determines cauchy distribution for F
        

    def mutatation(self):
        '''
        Generate a new mutant strategy based on the JADE mutation formula
        '''
        
        # pick four distinct strats at random
        stratlist = list(range(0, self.k))
        random.shuffle(stratlist)

        # s0 is next iteration's candidate for possible replacement
        self.diffevol['s0_index'] = stratlist[0]

        # s1, s2, s3 used in DE to create new strategy, potential replacement for s0
        i1, i2, i3 = stratlist[1], stratlist[2], stratlist[3]
        s1, s2, s3 = self.strats[i1]['stratval'], self.strats[i2]['stratval'], self.strats[i3]['stratval']

        s_best = 0

        # this is the differential evolution "adaptive step": create a new individual
        new_stratval = s1 + self.F * (s_best - s1) + self.F * (s2 - s3)

        # clip to bounds
        new_stratval = max(-1, min(+1, new_stratval))

        # record it for future use (s0 will be evaluated first, then s_new)
        self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval


    def crossover(self):
        pass

    def selection(self):
        pass

    def clean_archive(self):
        if self.archive > self.k:
            rand_strat = random.randint(0, self.k - 1)
            self.archive.pop(rand_strat)
    
    def lehmer_mean(self):
        sum_sqr = sum([f ** 2 for f in self.display])
        sum = sum(self.display)

        return sum_sqr / sum

    def generate_F(self):
        while True:
            val = cauchy.rvs(self.muF, 0.1)
            if val > 0:
                break
        
        self.F = min(val, 1)
        

    def update_params(self):
        self.clean_archive()


    def respond(self, time, lob, trade, verbose):
        # adaptive differential evolution

        # ensure k >= 4
        if self.k < 4:
            sys.exit('FAIL: k too small for diffevol')

        self.update_params() # ensure archive len < k

        # only initiate diff-evol once the active strat has been evaluated for long enough
        actv_lifetime = time - self.strats[self.active_strat]['start_t']
        if actv_lifetime >= self.strat_wait_time:

            if self.diffevol['de_state'] == 'active_s0':
                # s0 evaluated, next evaluate s_new
                self.active_strat = self.diffevol['snew_index']
                self.strats[self.active_strat]['start_t'] = time
                self.strats[self.active_strat]['profit'] = 0.0
                self.strats[self.active_strat]['pps'] = 0.0
                self.diffevol['de_state'] = 'active_snew'

            elif self.diffevol['de_state'] == 'active_snew':
                # evaluated both s0 and s_new, complete DE
                i_0 = self.diffevol['s0_index']
                i_new = self.diffevol['snew_index']

                # check for highest pps between s0 and s_new
                if self.strats[i_new]['pps'] >= self.strats[i_0]['pps']:
                    # archive unsuccessful strat
                    self.archive.append(self.strats[i_0]['stratval'])

                    # overwrite s0
                    self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

                    # store successful F value
                    self.display.append(self.F)

                # ADAPTIVE DIFFERENTIAL EVOLUTION

                self.generate_F()

                self.mutation()










                # DC's intervention for fully converged populations
                # is the stddev of the strategies in the population equal/close to zero?
                sum = 0.0
                for s in range(self.k):
                    sum += self.strats[s]['stratval']
                strat_mean = sum / self.k
                sumsq = 0.0
                for s in range(self.k):
                    diff = self.strats[s]['stratval'] - strat_mean
                    sumsq += (diff * diff)
                strat_stdev = math.sqrt(sumsq / self.k)
                
                if strat_stdev < 0.0001:
                    # this population has converged
                    # mutate one strategy at random
                    randindex = random.randint(0, self.k - 1)
                    self.strats[randindex]['stratval'] = random.uniform(-1.0, +1.0)

                # set up next iteration: first evaluate s0
                self.active_strat = self.diffevol['s0_index']
                self.strats[self.active_strat]['start_t'] = time
                self.strats[self.active_strat]['profit'] = 0.0
                self.strats[self.active_strat]['pps'] = 0.0

                self.diffevol['de_state'] = 'active_s0'

            else:
                sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')