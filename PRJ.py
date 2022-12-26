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
        self.p = params['p'] # top 100p% for best strat
        self.muF = 0.5 # determines cauchy distribution for F
        

    def mutation(self):
        '''
        Generate a new mutant strategy based on the JADE mutation formula
        '''
        # order strats by pps descending
        ordered = sorted(self.strats, key=lambda x: self.strats[x]['pps'], reverse=True)
        quantile = ordered[: int(self.p * self.k)] # select top p-quantile
        random.shuffle(quantile)
        s_best = quantile[0] # top 100p% chosen at random

        # pick k distinct strategies at random
        stratlist = list(range(0, self.k))
        random.shuffle(stratlist)

        # s0 is next iteration's candidate for possible replacement
        self.diffevol['s0_index'] = stratlist[0]

        s0 = self.strats[self.diffevol['s0_index']]['stratval']
        s1 = stratlist[1] # randomly choose from current population

        intersection = [self.strats[i]['stratval'] for i in stratlist[2:]] + self.archive
        random.shuffle(intersection)
        s2 = intersection[0] # randomly choose from current population intersect archive

        # differential evolution mutation
        new_stratval = s0 + self.F * (s_best - s0) + self.F * (s1 - s2)
        new_stratval = max(-1, min(1, new_stratval)) # clip to bounds

        # record it for future use (s0 will be evaluated first, then s_new)
        self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval


    def selection(self):
        '''
        Select the most profitable strat between s0 and s_new
        '''

        i_0 = self.diffevol['s0_index']
        i_new = self.diffevol['snew_index']

        if self.strats[i_new]['pps'] >= self.strats[i_0]['pps']:
            # archive unsuccessful strat
            self.archive.append(self.strats[i_0]['stratval'])

            # overwrite s0
            self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

            # store successful F value
            self.display.append(self.F)

    def clean_archive(self):
        '''
        Remove strats from archive if length exceeds k
        '''
        if len(self.archive) > self.k:
            random.shuffle(self.archive)
            self.archive = self.archive[:self.k]
    
    def lehmer_mean(self):
        '''
        Calculate the Lehmer mean for updating muF
        '''
        sum_sqr = sum([f ** 2 for f in self.display])
        sum = sum(self.display)

        return sum_sqr / (sum if sum else 0.0001)

    def generate_F(self):
        '''
        Generate a new DE weight using Cauchy distribution
        '''
        while True:
            val = cauchy.rvs(self.muF, 0.1)
            if val > 0:
                break
        
        self.F = min(val, 1)
        

    def check_strat_convergence(self):
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

    def update_params(self):
        self.clean_archive() # ensure archive len < k
        self.display = []

        # update cauchy mean
        self.muF = (1 - self.c) * self.muF + self.c * self.lehmer_mean()


    def respond(self, time, lob, trade, verbose):
        # adaptive differential evolution

        # ensure k >= 4
        if self.k < 4:
            sys.exit('FAIL: k too small for diffevol')

        # only initiate diff-evol once the active strat has been evaluated for long enough
        actv_lifetime = time - self.strats[self.active_strat]['start_t']
        if actv_lifetime >= self.strat_wait_time:
            
            # s0 evaluated, next evaluate s_new
            if self.diffevol['de_state'] == 'active_s0':
                
                self.active_strat = self.diffevol['snew_index']
                self.strats[self.active_strat]['start_t'] = time
                self.strats[self.active_strat]['profit'] = 0.0
                self.strats[self.active_strat]['pps'] = 0.0
                self.diffevol['de_state'] = 'active_snew'

            # evaluated both s0 and s_new, complete DE algorithm
            elif self.diffevol['de_state'] == 'active_snew':
                
                self.generate_F()
                
                # ADAPTIVE DIFFERENTIAL EVOLUTION (JADE)
                self.selection()
                self.mutation()

                # set up next iteration: first evaluate s0
                self.active_strat = self.diffevol['s0_index']
                self.strats[self.active_strat]['start_t'] = time
                self.strats[self.active_strat]['profit'] = 0.0
                self.strats[self.active_strat]['pps'] = 0.0

                self.diffevol['de_state'] = 'active_s0'

                # reset for next generation
                self.update_params()

            else:
                sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')