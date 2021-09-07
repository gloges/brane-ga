import numpy as np
import os
from population import population

import matplotlib.pyplot as plt




def GA(envParams, GAmetaParams, xoverProbs, mutRates, fitnessParams,
       displayRun, saveTKS, paramScanString):


    env, bix2, UI, kMin, kMax  = envParams
    nRuns, gMax, nPop, nElite = GAmetaParams

    TKSstringsALL = ['', 'T', 'K', 'S', 'TK', 'TS', 'KS', 'TKS']
    TKSstrings = ['TK', 'TS', 'KS', 'TKS']


    if saveTKS:
        folderTKS, fileNameTKS = createFolders(envParams, GAmetaParams, paramScanString)



    TKSmodels = [[np.empty([0, k, 7], dtype='int') for k in range(kMax+1)] for s in TKSstrings]


    for n in range(nRuns):
        print('Starting run %d...' % (n+1))



        # Initialize population
        pop = population(nPop, envParams, fitnessParams)

        if displayRun:
            fittestRewards = np.zeros([gMax + 1, 4])
            aveRewards = np.zeros([gMax + 1, 4])
            fitnessQuantiles = np.zeros([gMax + 1, 4])
            braneTypes = np.zeros([gMax + 1, 8])
            TKSCounts = np.zeros([gMax + 1, len(TKSstringsALL)])

            fittestRewards[0] = pop.getFittest(1)[0].rewards
            aveRewards[0] = pop.getAveRewards()
            fitnessQuantiles[0] = pop.getFitnessQuantiles()
            braneTypes[0] = pop.getAveBraneTypes()
            TKSCounts[0] = pop.getTKSCounts(TKSstringsALL)


        for g in range(1, gMax+1):

            pop.breed(nElite, xoverProbs, mutRates)


            # Collect T/K/S models
            TKSnew = pop.getTKS(TKSstrings)
            for i in range(len(TKSstrings)):
                for ind in TKSnew[i]:
                    k = len(ind.chromosome)
                    TKSmodels[i][k] = np.append(TKSmodels[i][k], [ind.saveSort()], axis=0)

                # Remove duplicates
                for k in range(kMax+1):
                    if len(TKSmodels[i][k]) > 0:
                        TKSmodels[i][k] = np.unique(TKSmodels[i][k], axis=0)

            if displayRun:
                # Collect info
                fittestRewards[g] = pop.getFittest(1)[0].rewards
                aveRewards[g] = pop.getAveRewards()
                fitnessQuantiles[g] = pop.getFitnessQuantiles()
                braneTypes[g] = pop.getAveBraneTypes()
                TKSCounts[g] = pop.getTKSCounts(TKSstringsALL)



        if displayRun:
            fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True)

            ax[0].plot(fittestRewards[:, 0], label='T: best')
            ax[0].plot(fittestRewards[:, 1], label='K: best')
            ax[0].plot(fittestRewards[:, 2], label='S: best')
            ax[0].plot(fittestRewards[:, 3], label='MSSM: best')
            ax[0].plot(aveRewards[:, 0], '--', c='C0', label='T: ave')
            ax[0].plot(aveRewards[:, 1], '--', c='C1', label='K: ave')
            ax[0].plot(aveRewards[:, 2], '--', c='C2', label='S: ave')
            ax[0].plot(aveRewards[:, 3], '--', c='C3', label='MSSM: ave')
            ax[0].legend()
            ax[0].set_ylim([-0.05, 1.05])

            ax[1].plot(fitnessQuantiles[:, 0], label='Q25')
            ax[1].plot(fitnessQuantiles[:, 1], label='Q50')
            ax[1].plot(fitnessQuantiles[:, 2], label='Q75')
            ax[1].plot(fitnessQuantiles[:, 3], label='Q100')
            ax[1].legend()
            ax[1].set_ylim([-0.05, 1.05])

            ax[2].stackplot(np.arange(gMax+1), braneTypes.T)
            ax[2].legend(['A', 'B', 'C', "A'", "B'", "C'", "D'", "E'"])

            ax[3].stackplot(np.arange(gMax+1), TKSCounts.T)
            ax[3].legend(TKSstringsALL)

            plt.xlim([-2, gMax + 10])
            plt.show()


        for ind in pop.getFittest(1):
            ind.display()


        # Print number of consistent solutions found
        print('\tk   | ', end='')
        for k in range(kMin, kMax + 1):
            print('%6d' % k, end='')
        print(' | total\n\t' + '-'*(20+6*(kMax-kMin)))
        for i in range(len(TKSstrings)):
            print('\t%-3s | ' % TKSstrings[i], end='')
            for k in range(kMin, kMax + 1):
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


        if saveTKS:

            for i in range(len(TKSstrings)):

                for k in range(kMax + 1):

                    if len(TKSmodels[i][k]) > 0:

                        # Create filepath
                        file = folderTKS + '/' + TKSstrings[i] + ("_k=%d_" % k) + fileNameTKS

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


def createFolders(envParams, GAmetaParams, paramScanString):
    # Creates file system for saving models

    env, bix2, UI, kMin, kMax  = envParams
    nRuns, gMax, nPop, nElite = GAmetaParams

    rnd = ''.join([str(d) for d in np.random.randint(10, size=10)])


    folderbix2 = 'models/bix2= {} {} {}'.format(bix2[0], bix2[1], bix2[2])

    folderUI = folderbix2 + '/UI= {} {} {} {}'.format(UI[0], UI[1], UI[2], UI[3])

    fileNameTKS = 'env={}_nRuns={}_gMax={}_'.format(env, nRuns, gMax)
    fileNameTKS += 'nPop={}_nElite={}{}_{}.npy'.format(nPop, nElite, paramScanString, rnd)

    try:
        os.mkdir(folderbix2)
    except IOError as err:
        print(err)

    try:
        os.mkdir(folderUI)
    except IOError as err:
        print(err)

    return folderUI, fileNameTKS
