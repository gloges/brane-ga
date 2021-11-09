import numpy as np
from individual import individual


class population:

    # Initialize to random population of individuals
    def __init__(self, nPop, env, bix2, UI, kMin, kMax, NMax, weights, tadScale, susyScale):

        self.nPop = nPop
        self.env = env
        self.bix2 = bix2
        self.UI = UI
        self.kMin = kMin
        self.kMax = kMax
        self.NMax = NMax
        self.weights = weights
        self.tadScale = tadScale
        self.susyScale = susyScale

        # Initialize population of individuals
        self.individuals = np.empty(self.nPop, dtype=individual)

        i = 0
        while i < self.nPop:
            ind = individual(self.bix2, self.kMin, self.kMax)
            ind.adjust(self.env, self.NMax)
            ind.standardize()
            ind.updateFitness(self.env, self.UI, self.weights, self.tadScale, self.susyScale)

            if len(ind.chromosome) >= self.kMin:
                self.individuals[i] = ind
                i += 1

    # Breed next generation (w/ elitism)
    def breed(self, nElite, xoverProbs, mutRates):

        # Create array for next generation and copy over fittest individuals
        newIndividuals = np.empty([self.nPop], dtype=individual)
        newIndividuals[:nElite] = self.getFittest(nElite)

        # Create individuals via cross-over/mutation until next generation is full
        i = nElite
        while i < self.nPop:

            # Select two parents
            parent1 = self.binaryTournament()
            parent2 = self.binaryTournament()

            # Generate child by cross-over
            child = parent1.crossOver(parent2, xoverProbs)

            # Apply mutations, adjust according to environment, the set to standard form
            child.mutate(mutRates)
            child.adjust(self.env, self.NMax)
            child.standardize()

            # Check child contains enough stacks (some may have been removed in 'adjust' step)
            if len(child.chromosome) >= self.kMin:
                # Add child to next generation
                newIndividuals[i] = child
                i += 1

        # Update population to newly created individuals and compute all fitnesses
        self.individuals = newIndividuals
        for ind in self.individuals:
            ind.updateFitness(self.env, self.UI, self.weights, self.tadScale, self.susyScale)

    # From two individuals return the fittest
    def binaryTournament(self):
        # Select two random individuals
        i, j = np.random.randint(self.nPop, size=2)

        if self.individuals[i].fitness < self.individuals[j].fitness:
            return self.individuals[i]
        else:
            return self.individuals[j]

    # Returns the n fittest individuals
    def getFittest(self, n):

        # Get all fitnesses and sort
        fitnesses = [e.fitness for e in self.individuals]
        bestn = np.argsort(fitnesses)[:n]

        return self.individuals[bestn]

    # Compute
    def getAveFitnessTerms(self):
        return np.mean([ind.fitnessTerms for ind in self.individuals], axis=0)

    # Returns fitness 25/50/75/100 quantiles
    def getFitnessQuantiles(self):
        fitnesses = [ind.fitness for ind in self.individuals]

        Q00 = np.min(fitnesses)
        Q25 = np.quantile(fitnesses, 0.25)
        Q50 = np.quantile(fitnesses, 0.50)
        Q75 = np.quantile(fitnesses, 0.75)

        return [Q00, Q25, Q50, Q75]

    def getAveBraneTypes(self):

        return np.mean([ind.braneTypes()[1] for ind in self.individuals], axis=0)

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
