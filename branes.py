import numpy as np
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
    def run(self, nRuns, returnData=False, optimization=False,
            saveTKS=False, TKStosave=TKS_STRINGS, paramString=''):

        if optimization or saveTKS:
            self.createFolders(nRuns, paramString)

        allFits = np.zeros([self.gMax + 1, self.nPop])
        allFitTerms = np.zeros([self.gMax + 1, self.nPop, 4])

        for n in range(nRuns):
            # Initialize population
            pop = population(self. nPop, self.env, self.bix2, self.UI, self.kMin, self.kMax,
                             self.NMax, self.weights, self.tadScale, self.susyScale)

            # Empty array to store TK/TS/KS/TKS models
            TKSmodels = [[np.empty([0, k, 7], dtype='int') for k in range(self.kMax+1)]
                         for s in TKS_STRINGS]

            allFits[0] = pop.getAllFitnesses()
            allFitTerms[0] = pop.getAllFitTerms()

            # Initialize arrays for summary of GA run
            fitnessQuantiles = np.zeros([self.gMax + 1, 4])
            braneTypes = np.zeros([self.gMax + 1, 8])
            TKSCounts = np.zeros([self.gMax + 1, len(TKS_STRINGS_ALL)])
            TKSunique = np.zeros(self.gMax + 1)

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

                allFits[g] = pop.getAllFitnesses()
                allFitTerms[g] = pop.getAllFitTerms()

                # Collect summary info for current generation
                fitnessQuantiles[g] = pop.getFitnessQuantiles()
                braneTypes[g] = pop.getAveBraneTypes()
                TKSCounts[g] = pop.getTKSCounts(TKS_STRINGS_ALL)
                TKSunique[g] = sum([len(TKSmodels[-1][k]) for k in range(self.kMax+1)])

            # Display fittest individual in population
            for ind in pop.getFittest(1):
                ind.display()

            self.displayTKScounts(TKSmodels)

            if optimization or saveTKS:
                self.saveTKStoFile(TKSmodels, TKStosave)

            if returnData:
                return allFits, allFitTerms, fitnessQuantiles, braneTypes, TKSCounts, TKSunique

        if optimization:
            # Collect all files containing unique TKSmodels and count
            counts = np.zeros(self.kMax+1, dtype='int')

            for k in range(self.kMin, self.kMax+1):
                files = glob.glob(('models/bix2= 0 0 0/UI= 1 1 1 1/TKS_k=%d_' % k)+self.fileNameTKS)

                if len(files) > 0:
                    # There should only be one such file
                    counts[k] = len(np.load(files[0]))

            try:
                # Load previously saved
                loaded = np.load('models/bix2= 0 0 0/UI= 1 1 1 1/TKS_counts_' + self.fileNameTKS)
                toSave = loaded
                toSave[-1] += counts
            except IOError:
                # No such file exists
                toSave = [self.xoverProbs, self.mutRates, self.weights,
                          [self.tadScale, self.susyScale], counts]

            np.save('models/bix2= 0 0 0/UI= 1 1 1 1/TKS_counts_' + self.fileNameTKS, toSave)

            for file in glob.glob('models/*/*/*_k=*.npy'):
                os.remove(file)

    # Creates file system for saving models
    def createFolders(self, nRuns, paramString):

        # Ten random digits to ensure unique filename
        rnd = ''.join([str(d) for d in np.random.randint(10, size=10)])

        folderbix2 = 'models/bix2= {} {} {}'.format(*self.bix2)

        folderTKS = folderbix2 + '/UI= {} {} {} {}'.format(*self.UI)

        fileNameTKS = 'env={}_nRuns={}_gMax={}_'.format(self.env, nRuns, self.gMax)
        fileNameTKS += 'nPop={}_nElite={}{}_{}.npy'.format(self.nPop, self.nElite,
                                                           paramString, rnd)

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


# Analyze previously saved models
def getStatistics(filePath, printQ=True):

    # Collect all files and load data
    files = np.array(glob.glob(filePath))
    if printQ:
        print('{} files found...'.format(len(files)))

    if len(files) == 0:
        if printQ:
            print('Nothing to do...')
        return None

    # Extract info about environment/run from filepaths
    bix2Data = np.array([[int(b) for b in f.split('\\')[2].split(' ')[1:]] for f in files])
    UIData = np.array([[int(U) for U in f.split('\\')[3].split(' ')[1:]] for f in files])
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
    Tmeans = np.empty([0], dtype='float')
    Kmeans = np.empty([0], dtype='float')
    Smeans = np.empty([0], dtype='float')
    ranks = np.empty([0], dtype='int')
    meanChiralities = np.empty([0], dtype='float')
    ABCs  = np.empty([0, 3], dtype='int')
    SUNs  = np.empty([0, 11], dtype='int')
    Qs = np.empty([0, 3], dtype='int')
    Ls = np.empty([0, 2], dtype='int')
    U1massless = np.empty([0], dtype='int')
    alphas = np.empty([0, 3], dtype='float')

    headers = ['env', 'bix2', 'UI', 'T/K/S', 'k', 'models', 'unique', 'repeats']
    if printQ:
        print('Gathering statistics...')
        print('{:>5}{:>11}{:>17}{:>7}{:>5}{:>10}{:>10}{:>10}'.format(*headers))
        print(77*'-')

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
                UIinds *= (UIData[:, 3] == UI[3])

                for TKSlabel in TKSunique:
                    TKSinds = (TKSData == TKSlabel)

                    for k in kUnique:
                        kinds = (kData == k)

                        mask = envinds * bix2inds * UIinds * TKSinds * kinds

                        models = np.empty([0, k, 7], dtype='int')

                        if sum(mask) > 0:
                            if printQ:
                                print('{:>5}'.format(env), end='')
                                print(4*' ' + '({},{},{})'.format(*bix2), end='')
                                print(4*' ' + '({:>2},{:>2},{:>2},{:>2})'.format(*UI), end='')
                                print(4*' ' + TKSlabel, end='')
                                print(4*' ' + str(k), end='')

                            for f in files[mask]:
                                models = np.append(models, np.load(f), axis=0)

                            if printQ:
                                print('{:10,}'.format(len(models)), end='')
                            # At this point 'models' contains all those models with
                            # the same meta-parameters. Now restrict to unique models
                            # and compute all statistics.

                            modelsUnique = np.unique(models, axis=0)

                            if printQ:
                                print('{:10,}'.format(len(modelsUnique)), end='')
                                print('{:>10.1%}'.format(1 - len(modelsUnique)/len(models)))

                            for model in modelsUnique:

                                # Get information about this model:
                                # <T>, <K>, <S>, rank, ABCcounts, SU(N)counts, etc.
                                info = individual.getInfo(model, env, bix2, UI, 10)

                                if info[0] <= 8:
                                    # Add to statistics arrays
                                    envs  = np.append(envs, [env])
                                    bix2s = np.append(bix2s, [bix2], axis=0)
                                    UIs   = np.append(UIs, [UI], axis=0)
                                    TKSs  = np.append(TKSs, [TKSlabel])
                                    ks    = np.append(ks, [k])
                                    Tmeans = np.append(Tmeans, [info[0]])
                                    Kmeans = np.append(Kmeans, [info[1]])
                                    Smeans = np.append(Smeans, [info[2]])
                                    ranks = np.append(ranks, [info[3]])
                                    meanChiralities = np.append(meanChiralities, [info[4]])
                                    ABCs  = np.append(ABCs, [info[5]], axis=0)
                                    SUNs  = np.append(SUNs, [info[6]], axis=0)
                                    Qs = np.append(Qs, [info[7]], axis=0)
                                    Ls = np.append(Ls, [info[8]], axis=0)
                                    U1massless = np.append(U1massless, [info[9]], axis=0)
                                    alphas = np.append(alphas, [info[10]], axis=0)

    return envs, bix2s, UIs, TKSs, ks, Tmeans, Kmeans, Smeans, \
        ranks, meanChiralities, ABCs, SUNs, Qs, Ls, U1massless, alphas
