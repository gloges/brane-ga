import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from population import population
import individual

"""


"""

TKS_STRINGS_ALL = ['', 'T', 'K', 'S', 'TK', 'TS', 'KS', 'TKS']
TKS_STRINGS = ['TK', 'TS', 'KS', 'TKS']


class braneGA:

    def setEnvParams(self, env, bix2, UI, kMin, kMax, NMax):

        self.env = env
        self.bix2 = bix2
        self.UI = UI
        self.kMin = kMin
        self.kMax = kMax
        self.NMax = NMax

    def setGAparams(self, gMax, nPop, nElite, xoverProbs, mutRates, weights, tadScale, susyScale):

        self.gMax = gMax
        self.nPop = nPop
        self.nElite = nElite
        self.xoverProbs = xoverProbs
        self.mutRates = mutRates
        self.weights = weights
        self.tadScale = tadScale
        self.susyScale = susyScale

    # Run genetic algorithm 'nRuns' times
    def run(self, nRuns, displayRun, saveTKS, TKStosave=TKS_STRINGS, paramScanString=''):

        if saveTKS:
            self.createFolders(nRuns, paramScanString)

        for n in range(nRuns):
            # Initialize population
            pop = population(self. nPop, self.env, self.bix2, self.UI, self.kMin, self.kMax,
                             self.NMax, self.weights, self.tadScale, self.susyScale)

            # Empty array to store TK/TS/KS/TKS models
            TKSmodels = [[np.empty([0, k, 7], dtype='int') for k in range(self.kMax+1)]
                         for s in TKS_STRINGS]

            # Initialize arrays for summary of GA run
            if displayRun:
                bestFitnessTerms = np.zeros([self.gMax + 1, 4])
                aveFitnessTerms = np.zeros([self.gMax + 1, 4])
                fitnessQuantiles = np.zeros([self.gMax + 1, 4])
                braneTypes = np.zeros([self.gMax + 1, 8])
                TKSCounts = np.zeros([self.gMax + 1, len(TKS_STRINGS_ALL)])
                TKSunique = np.zeros(self.gMax + 1)

                bestFitnessTerms[0] = pop.getFittest(1)[0].fitnessTerms
                aveFitnessTerms[0] = pop.getAveFitnessTerms()
                fitnessQuantiles[0] = pop.getFitnessQuantiles()
                braneTypes[0] = pop.getAveBraneTypes()
                TKSCounts[0] = pop.getTKSCounts(TKS_STRINGS_ALL)
                TKSunique[0] = 0

            # Loop through gMax generations
            for g in range(1, self.gMax+1):

                pop.breed(self.nElite, self.xoverProbs, self.mutRates)

                # Collect TK/TS/KS/TKS models
                TKSnew = pop.getTKS(TKS_STRINGS)
                for i in range(len(TKS_STRINGS)):
                    for ind in TKSnew[i]:
                        k = len(ind.chromosome)
                        TKSmodels[i][k] = np.append(TKSmodels[i][k], [ind.saveSort()], axis=0)

                    # Remove duplicates
                    for k in range(self.kMin, self.kMax+1):
                        if len(TKSmodels[i][k]) > 0:
                            TKSmodels[i][k] = np.unique(TKSmodels[i][k], axis=0)

                # Collect summary info for current generation
                if displayRun:
                    bestFitnessTerms[g] = pop.getFittest(1)[0].fitnessTerms
                    aveFitnessTerms[g] = pop.getAveFitnessTerms()
                    fitnessQuantiles[g] = pop.getFitnessQuantiles()
                    braneTypes[g] = pop.getAveBraneTypes()
                    TKSCounts[g] = pop.getTKSCounts(TKS_STRINGS_ALL)
                    TKSunique[g] = sum([len(TKSmodels[-1][k]) for k in range(self.kMax+1)])

            if displayRun:
                createRunPlot(self.gMax, bestFitnessTerms, aveFitnessTerms,
                              fitnessQuantiles, braneTypes, TKSCounts, TKSunique)

            # Display fittest individual in population
            for ind in pop.getFittest(1):
                ind.display()

            self.displayTKScounts(TKSmodels)

            if saveTKS:
                self.saveTKStoFile(TKSmodels, TKStosave)

    # Creates file system for saving models
    def createFolders(self, nRuns, paramScanString):

        # Ten random digits to ensure unique filename
        rnd = ''.join([str(d) for d in np.random.randint(10, size=10)])

        folderbix2 = 'models/bix2= {} {} {}'.format(*self.bix2)

        folderTKS = folderbix2 + '/UI= {} {} {} {}'.format(*self.UI)

        fileNameTKS = 'env={}_nRuns={}_gMax={}_'.format(self.env, nRuns, self.gMax)
        fileNameTKS += 'nPop={}_nElite={}{}_{}.npy'.format(self.nPop, self.nElite,
                                                           paramScanString, rnd)

        try:
            os.mkdir(folderbix2)
        except IOError as err:
            print(err)

        try:
            os.mkdir(folderTKS)
        except IOError as err:
            print(err)

        self.folderTKS = folderTKS
        self.fileNameTKS = fileNameTKS

    # Save TKSmodels to file while removing duplicates
    def saveTKStoFile(self, TKSmodels, TKStosave):

        # Loop through TK/TS/KS/TKS
        for i in range(len(TKS_STRINGS)):

            if TKS_STRINGS[i] not in TKStosave:
                continue

            # Loop through numbers of stacks
            for k in range(self.kMin, self.kMax + 1):

                if len(TKSmodels[i][k]) > 0:

                    # Create filepath
                    file = self.folderTKS + '/' + TKS_STRINGS[i] \
                        + ("_k=%d_" % k) + self.fileNameTKS

                    try:
                        # Load previously saved
                        loaded = np.load(file)
                        toSave = np.append(loaded, TKSmodels[i][k], axis=0)
                    except IOError:
                        # No such file exists
                        toSave = TKSmodels[i][k]

                    # Remove duplicates
                    toSave = np.unique(toSave, axis=0)

                    # Save to file
                    np.save(file, toSave)

    def displayTKScounts(self, TKSmodels):

        # Print number of consistent solutions
        print('\tk   | ', end='')
        for k in range(self.kMin, self.kMax + 1):
            print('%6d' % k, end='')
        print(' | total\n\t' + '-'*(20+6*(self.kMax-self.kMin)))
        for i in range(len(TKS_STRINGS)):
            print('\t%-3s | ' % TKS_STRINGS[i], end='')
            for k in range(self.kMin, self.kMax + 1):
                a = len(TKSmodels[i][k])
                if a > 0:
                    print('%6d' % a, end='')
                else:
                    print(' '*6, end='')
            a = sum([len(c) for c in TKSmodels[i]])
            if a > 0:
                print(' |%6d' % a)
            else:
                print(' |' + ' '*6)
        print('\n\n')

# Summary plot of single GA run
def createRunPlot(gMax, bestFitnessTerms, aveFitnessTerms,
                  fitnessQuantiles, braneTypes, TKSCounts, TKSunique):

    fig, ax = plt.subplots(5, 1, figsize=(13, 20), sharex=True)

    ax[0].plot(bestFitnessTerms[:, 0], label='T: best')
    ax[0].plot(bestFitnessTerms[:, 1], label='K: best')
    ax[0].plot(bestFitnessTerms[:, 2], label='S: best')
    ax[0].plot(bestFitnessTerms[:, 3], label='MSSM: best')
    ax[0].plot(aveFitnessTerms[:, 0], '--', c='C0', label='T: ave')
    ax[0].plot(aveFitnessTerms[:, 1], '--', c='C1', label='K: ave')
    ax[0].plot(aveFitnessTerms[:, 2], '--', c='C2', label='S: ave')
    ax[0].plot(aveFitnessTerms[:, 3], '--', c='C3', label='MSSM: ave')
    ax[0].legend()
    ax[0].set_ylim([-0.05, 1.05])

    ax[1].plot(fitnessQuantiles[:, 0], label='Q00')
    ax[1].plot(fitnessQuantiles[:, 1], label='Q25')
    ax[1].plot(fitnessQuantiles[:, 2], label='Q50')
    ax[1].plot(fitnessQuantiles[:, 3], label='Q75')
    ax[1].legend()
    ax[1].set_ylim([-0.05, 1.05])

    ax[2].stackplot(np.arange(gMax+1), braneTypes.T)
    ax[2].legend(['A', 'B', 'C', "A'", "B'", "C'", "D'", "E'"])

    ax[3].stackplot(np.arange(gMax+1), TKSCounts.T)
    ax[3].legend(TKS_STRINGS_ALL)

    ax[4].plot(TKSunique)

    plt.xlim([-0.02*gMax, 1.1*gMax])
    plt.show()


# Analyze previously saved models
def getStatistics(filePath):

    # Collect all files and load data
    files = glob.glob(filePath)
    print('{} files found...'.format(len(files)))

    modelData = np.array([np.load(f) for f in files])
    print('File data loaded.')

    # Extract info about environment/run from filepaths
    bix2Data = np.array([[int(b) for b in f.split('\\')[1].split(' ')[1:]] for f in files])
    UIData = np.array([[int(U) for U in f.split('\\')[2].split(' ')[1:]] for f in files])
    metaData = np.array([f.split('\\')[-1].split('_') for f in files])
    TKSData = metaData[:, 0]
    kData = [int(a[1][2:]) for a in metaData]
    envData = [int(a[2][4:]) for a in metaData]

    bix2Unique = np.unique(bix2Data, axis=0)
    UIUnique = np.unique(UIData, axis=0)
    TKSunique = np.unique(TKSData, axis=0)
    kUnique = np.unique(kData, axis=0)
    envData = np.unique(envData, axis=0)

    # Prepare arrays for statistics
    envs  = np.empty([0], dtype='int')
    bix2s = np.empty([0, 3], dtype='int')
    UIs   = np.empty([0, 4], dtype='int')
    TKSs  = np.empty([0], dtype='str')
    ks    = np.empty([0], dtype='int')
    TL1s  = np.empty([0], dtype='int')
    KL1s  = np.empty([0], dtype='int')
    ranks = np.empty([0], dtype='int')
    ABCs  = np.empty([0, 3], dtype='int')
    SUNs  = np.empty([0, 11], dtype='int')

    print('Gathering statistics...')

    # Loop through unique combinations of meta-parameters
    for env in envData:
        envinds = (envData == env)

        for bix2 in bix2Unique:
            bix2inds = (bix2Data[:, 0] == bix2[0])
            bix2inds *= (bix2Data[:, 1] == bix2[1])
            bix2inds *= (bix2Data[:, 2] == bix2[2])

            for UI in UIUnique:
                UIinds = (UIData[:, 0] == UI[0])
                UIinds *= (UIData[:, 1] == UI[1])
                UIinds *= (UIData[:, 2] == UI[2])

                for TKSlabel in TKSunique:
                    TKSinds = (TKSData == TKSlabel)

                    for k in kUnique:
                        kinds = (kData == k)

                        print('\tenv={}'.format(env), end='')
                        print(', bix2=({},{},{})'.format(*bix2), end='')
                        print(', UI=({},{},{},{})'.format(*UI), end='')
                        print(', ' + TKSlabel + ', k={}: '.format(k))

                        mask = envinds * bix2inds * UIinds * TKSinds * kinds

                        modelsArray = modelData[mask]
                        models = np.empty([0, k, 7], dtype='int')

                        for mm in modelsArray:
                            models = np.append(models, mm, axis=0)

                        print('\t\tmodels: {:10,}'.format(len(models)))
                        # At this point 'models' contains all those models with
                        # the same meta-parameters. Now restrict to unique models
                        # and compute all statistics.

                        modelsUnique = np.unique(models, axis=0)

                        print('\t\tunique: {:10,}'.format(len(modelsUnique)))

                        for model in modelsUnique:

                            # Get information about this model:
                            # [|T|_1, sum(K), rank, ABCcounts, SU(N)counts]
                            info = individual.getInfo(model, env, bix2, UI, 10)

                            # Add to statistics arrays
                            envs  = np.append(envs, [env])
                            bix2s = np.append(bix2s, [bix2], axis=0)
                            UIs   = np.append(UIs, [UI], axis=0)
                            TKSs  = np.append(TKSs, [TKSlabel])
                            ks    = np.append(ks, [k])
                            TL1s  = np.append(TL1s, [info[0]])
                            KL1s  = np.append(KL1s, [info[1]])
                            ranks = np.append(ranks, [info[2]])
                            ABCs  = np.append(ABCs, [info[3]], axis=0)
                            SUNs  = np.append(SUNs, [info[4]], axis=0)

    return env, bix2s, UIs, TKSs, ks, TL1s, KL1s, ranks, ABCs, SUNs
