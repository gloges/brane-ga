import numpy as np
from individual import individual




class population:


    def __init__(self, nPop, envParams, fitnessParams):

        env, bix2, UI, kMin, kMax  = envParams

        self.nPop = nPop
        self.env = env
        self.UI = UI
        self.kMin = kMin
        self.fitnessParams = fitnessParams

        # Initialize population of individuals
        self.individuals = np.empty(nPop, dtype=individual)

        i = 0
        while i < nPop:
            ind = individual(bix2, kMin, kMax)
            ind.adjust(env)
            ind.standardize()
            ind.updateFitness(env, UI, fitnessParams)

            if len(ind.chromosome) >= kMin:
                self.individuals[i] = ind
                i += 1


    def breed(self, nElite, xoverProbs, mutRates):


        newIndividuals = np.empty([self.nPop], dtype=individual)
        newIndividuals[:nElite] = self.getFittest(nElite)


        i = nElite
        while i < self.nPop:

            # Select two parents
            parent1 = self.binaryTournament()
            parent2 = self.binaryTournament()

            # Generate child by cross-over
            child = parent1.crossOver(parent2, xoverProbs)

            # Apply mutations, adjust according to environment, the set to standard form
            child.mutate(mutRates)
            child.adjust(self.env)
            child.standardize()

            if len(child.chromosome) >= self.kMin:
                # Add child to next generation
                newIndividuals[i] = child
                i += 1

        # Update population to newly created individuals and compute all fitnesses
        self.individuals = newIndividuals
        for ind in self.individuals:
            ind.updateFitness(self.env, self.UI, self.fitnessParams)


    def binaryTournament(self):
        # Select two random individuals and return the fittest
        i, j = np.random.randint(self.nPop, size=2)

        if self.individuals[i].fitness > self.individuals[j].fitness:
            return self.individuals[i]
        else:
            return self.individuals[j]


    def getFittest(self, n):
        # Return the n fittest individuals

        fitnesses = [e.fitness for e in self.individuals]
        bestn = np.argsort(fitnesses)[-1:-(n+1):-1]

        return self.individuals[bestn]


    def getAveRewards(self):
        return np.mean([ind.rewards for ind in self.individuals], axis=0)


    def getFitnessQuantiles(self):
        fitnesses = [ind.fitness for ind in self.individuals]

        Q25 = np.quantile(fitnesses, 0.25)
        Q50 = np.quantile(fitnesses, 0.50)
        Q75 = np.quantile(fitnesses, 0.75)
        Qmax = np.max(fitnesses)

        return [Q25, Q50, Q75, Qmax]


    def getAveBraneTypes(self):

        return np.mean([ind.braneTypes() for ind in self.individuals], axis=0)


    def getTKSCounts(self, strings):
        # Return counts of how many individuals satisfy
        # different consistency condition combinations

        consConds = np.array([ind.consCond for ind in self.individuals])
        counts = [np.sum(consConds == s) for s in strings]

        return counts


    def getTKS(self, strings):
        # Return individuals which satisfy different T/K/S consistency condition combinations

        consConds = np.array([ind.consCond for ind in self.individuals])

        TKSmodels = [self.individuals[np.where(consConds == s)[0]] for s in strings]

        return TKSmodels
