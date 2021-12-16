import numpy as np


"""

An individual consists of an array of winding numbers and multiplicities
which represent a configuration of D6-branes.

"""

class individual:

    # Initialize individual to random winding numbers
    def __init__(self, bix2, kMin, kMax):

        self.bix2 = np.array(bix2)
        self.kMin = kMin
        self.kMax = kMax

        # Initialize chromosome to contain k stacks
        k = np.random.randint(self.kMin, self.kMax + 1)
        self.chromosome = np.array([randomStack(1/3, 10) for i in range(k)])

        self.fitness = None

    # Cross-over chromosome with that of parent2 using one of four methods
    def crossOver(self, parent2, xoverProbs):
        # xoverProbs is an array of four probabilities for the four
        # cross-over methods. All four methods use uniform, 1-point cross-overs
        # that ensure the child has between kMin and kMax stacks.

        # Randomly choose a cross-over method
        rnd = np.random.rand()
        cumulative = np.cumsum(xoverProbs)
        method = sum(cumulative < rnd)

        k1 = len(self.chromosome)
        k2 = len(parent2.chromosome)

        flatParent1 = self.chromosome.copy().ravel()
        flatParent2 = parent2.chromosome.copy().ravel()

        if method < 4:
            # All 1-point methods

            xoStack1 = np.inf
            xoStack2 = -np.inf

            if method == 0:
                # stack
                while xoStack1 + (k2-xoStack2) > self.kMax or xoStack1 + (k2-xoStack2) < self.kMin:
                    xoStack1 = np.random.randint(0, k1+1)
                    xoStack2 = np.random.randint(0, k2+1)
                xoGene = 0

            elif method == 1:
                # gene
                while xoStack1 + (k2-xoStack2) > self.kMax or xoStack1 + (k2-xoStack2) < self.kMin:
                    xoStack1 = np.random.randint(0, k1)
                    xoStack2 = np.random.randint(0, k2)
                xoGene = np.random.randint(1, 7)

            elif method == 2:
                # stackbar
                kGoal = np.random.choice([k1, k2])

                xoStack1 = np.random.randint(max(0, kGoal-k2), min(k1, kGoal)+1)
                xoStack2 = k2 - kGoal + xoStack1
                xoGene = 0

            elif method == 3:
                # genebar
                kGoal = np.random.choice([k1, k2])

                xoStack1 = np.random.randint(max(0, kGoal-k2), min(k1, kGoal))
                xoStack2 = k2 - kGoal + xoStack1
                xoGene = np.random.randint(1, 7)

            xoPt1 = 7*xoStack1 + xoGene
            xoPt2 = 7*xoStack2 + xoGene

            flatChild = flatParent1[:xoPt1]
            flatChild = np.append(flatChild, flatParent2[xoPt2:])

        elif method >= 4:
            # All 2-point methods

            xoStack1a = np.inf
            xoStack1b = -np.inf
            xoStack2a = -np.inf
            xoStack2b = np.inf

            if method == 4:
                # stack
                while xoStack1a + (k1-xoStack1b) + (xoStack2b-xoStack2a) > self.kMax \
                        or xoStack1a + (k1-xoStack1b) + (xoStack2b-xoStack2a) < self.kMin:
                    xoStack1a, xoStack1b = np.sort(np.random.randint(0, k1+1, size=2))
                    xoStack2a, xoStack2b = np.sort(np.random.randint(0, k2+1, size=2))
                xoGenea = 0
                xoGeneb = 0

            elif method == 5:
                # gene
                while xoStack1a + (k1-xoStack1b) + (xoStack2b-xoStack2a) > self.kMax \
                        or xoStack1a + (k1-xoStack1b) + (xoStack2b-xoStack2a) < self.kMin:
                    xoStack1a, xoStack1b = np.sort(np.random.randint(0, k1, size=2))
                    xoStack2a, xoStack2b = np.sort(np.random.randint(0, k2, size=2))
                xoGenea = np.random.randint(1, 7)
                xoGeneb = np.random.randint(1, 7)
                if xoStack1a == xoStack1b or xoStack2a == xoStack2b:
                    xoGenea, xoGeneb = np.sort([xoGenea, xoGeneb])

            elif method == 6:
                # stackbar
                kGoal = np.random.choice([k1, k2])

                kfrom1 = np.random.randint(max(0, kGoal-k2), min(k1, kGoal)+1)
                kfrom2 = kGoal - kfrom1
                xoStack1a = np.random.randint(0, kfrom1+1)
                xoStack1b = k1 - (kfrom1 - xoStack1a)
                xoStack2a = np.random.randint(0, k2-kfrom2+1)
                xoStack2b = xoStack2a + kfrom2
                xoGenea = 0
                xoGeneb = 0

            elif method == 7:
                # genebar
                kGoal = np.random.choice([k1, k2])

                kfrom1 = np.random.randint(max(1, kGoal-k2+1), min(k1, kGoal))
                kfrom2 = kGoal - kfrom1

                xoStack1a = np.random.randint(0, kfrom1)
                xoStack1b = k1 - (kfrom1 - xoStack1a)
                xoStack2a = np.random.randint(0, k2-kfrom2)
                xoStack2b = xoStack2a + kfrom2
                xoGenea = np.random.randint(1, 7)
                xoGeneb = np.random.randint(1, 7)
                if xoStack1a == xoStack1b or xoStack2a == xoStack2b:
                    xoGenea, xoGeneb = np.sort([xoGenea, xoGeneb])

            xoPt1a = 7*xoStack1a + xoGenea
            xoPt1b = 7*xoStack1b + xoGeneb
            xoPt2a = 7*xoStack2a + xoGenea
            xoPt2b = 7*xoStack2b + xoGeneb

            flatChild = flatParent1[:xoPt1a]
            flatChild = np.append(flatChild, flatParent2[xoPt2a:xoPt2b])
            flatChild = np.append(flatChild, flatParent1[xoPt1b:])

        kChild = int(len(flatChild) / 7)

        # Create child with newly created chromosome
        child = individual(self.bix2, self.kMin, self.kMax)
        child.chromosome = flatChild.reshape(kChild, 7)

        return child

    # Randomly apply mutations to individual's chromosome
    def mutate(self, mutRates):

        # Extract individual mutation rates
        mutNaPM, mutWindPM, mutWindSgns, mutS4 = mutRates

        # # Split stack with Na > 1 into two
        # if len(self.chromosome) < self.kMax and np.random.rand() < mutStackSplit:

        #     splittable = np.where(self.chromosome[:, 0] > 1)[0]
        #     if len(splittable) > 0:
        #         a = np.random.choice(splittable)
        #         b = np.random.randint(len(self.chromosome) + 1)
        #         Na = self.chromosome[a, 0]
        #         self.chromosome = np.insert(self.chromosome, b, self.chromosome[a], axis=0)
        #         self.chromosome[a, 0] = np.random.randint(1, Na)
        #         self.chromosome[b, 0] = Na - self.chromosome[a, 0]

        # Mutations which are applied to each stack
        for stack in self.chromosome:

            # Change Na by +/-1
            if np.random.rand() < mutNaPM / len(self.chromosome):
                stack[0] += np.random.choice([-1, 1])

            # Loop through stack's winding numbers
            for i in range(1, 7):
                if np.random.rand() < mutWindPM / (6*len(self.chromosome)):
                    stack[i] += np.random.choice([-2, -1, 1, 2])

            # Randomly assign new sign pattern
            if np.random.rand() < mutWindSgns / len(self.chromosome):
                # Loop over winding pairs
                for i in range(3):
                    n = stack[2*i+1]
                    m = stack[2*i+2]

                    rnd = np.random.rand()

                    if rnd < 0.25:
                        stack[2*i+1] = -n
                    elif rnd < 0.5:
                        stack[2*i+2] = -m - self.bix2[i] * n
                    elif rnd < 0.75:
                        stack[2*i+1] = -n
                        stack[2*i+2] = -m

            # Permute X,Y
            if np.random.rand() < mutS4 / len(self.chromosome):
                newStack = np.zeros(7)
                newStack[0] = stack[0]

                # Apply element of V = Z2 x Z2
                if np.random.rand() < 0.75:

                    ii = np.random.randint(3)

                    for i in range(3):
                        n = stack[2*i+1]
                        m = stack[2*i+2]

                        if i == ii:
                            newStack[2*i+1] = -n
                            newStack[2*i+2] = -m
                        else:
                            newStack[2*i+1] = m + self.bix2[i] * (n + m)
                            newStack[2*i+2] = -n - self.bix2[i] * m

                stack = newStack

                # Apply element of S3
                sigma = np.random.permutation(3)

                for i in range(3):
                    n = stack[2*sigma[i]+1]
                    m = stack[2*sigma[i]+2]

                    if self.bix2[i] == self.bix2[sigma[i]]:
                        newStack[2*i+1] = n
                        newStack[2*i+2] = m
                    elif self.bix2[i] == 0:
                        newStack[2*i+1] = n
                        newStack[2*i+2] = 2*m + n
                    else:
                        newStack[2*i+1] = n
                        if (m - n) % 2 == 0:
                            newStack[2*i+2] = (m - n) / 2
                        else:
                            newStack[2*i+2] = (m - n + np.random.choice([-1, 1])) / 2

                stack = newStack

        # # Permute stacks
        # if np.random.rand() < mutStackPerm:
        #     np.random.shuffle(self.chromosome)

    # Ensure that the chromosome complies with the restrictions of the environment
    def adjust(self, env, NMax):
        # The value of env = 1,2,3 gives which types of branes are allowed
        #  - env = 2,3 replace types A',B',C' branes with types A,B,C
        #  - env = 3 then removes all type C branes (but then adds
        #          them by hand for evaluating the fitness function)

        # Correct non-coprime winding number pairs.
        for stack in self.chromosome:

            # Ensure 1 <= Na <= NMax
            stack[0] = max(abs(stack[0]), 1)
            stack[0] = min(stack[0], NMax)

            # Loop over winding pairs
            for i in range(3):
                n = stack[2*i+1]
                m = stack[2*i+2]

                if n == 0 and m == 0:
                    # (0,0) ==> (1,0), (-1,0), (0,1) or (0,-1)
                    if np.random.rand() < 0.5:
                        stack[2*i+1] = np.random.choice([-1, 1])
                    else:
                        stack[2*i+2] = np.random.choice([-1, 1])

                elif n == 0 and abs(m) > 1:
                    # (0,m), |m|>1 ==> (1,m), (-1,m) or (0,sgn(m))
                    if np.random.rand() < 0.5:
                        stack[2*i+2] /= np.abs(stack[2*i+2])
                    else:
                        stack[2*i+1] = np.random.choice([-1, 1])

                elif abs(n) > 1 and m == 0:
                    # (n,0), |n|>1 ==> (n,1), (n,-1) or (sgn(n),0)
                    if np.random.rand() < 0.5:
                        stack[2*i+1] /= np.abs(stack[2*i+1])
                    else:
                        stack[2*i+2] = np.random.choice([-1, 1])

                elif np.gcd(n, m) > 1:
                    # (n,m) with n,m not coprime. Pick either n,m and
                    # increment or decrement until n,m are coprime
                    direction = np.random.choice([-1, 1])
                    index = 2*i + np.random.choice([1, 2])
                    while np.gcd(stack[2*i+1], stack[2*i+2]) > 1:
                        stack[index] += direction

        # Correct type A',B',C' branes
        if (env == 2) or (env == 3):

            for stack in self.chromosome:

                XI, YI = getStackXY(stack, self.bix2)

                # Count how many XI/YI are zero/positive
                nZeroXI = np.sum(XI == 0)
                nZeroYI = np.sum(YI == 0)
                nPosXI = np.sum(XI > 0)

                if (nZeroYI == 0 and nPosXI == 1) \
                   or (nZeroXI == 2 and nPosXI < 2) \
                   or (nZeroXI == 3 and nZeroYI == 4 and nPosXI == 0):
                    # Type A', B', C'

                    # Some type B' cases require special treatment
                    if nZeroXI == 2 and nPosXI == 1:

                        if XI[0] == 0:
                            # If X0=0, flip mhat^i corresponding to X^i>0
                            iPos = np.where(XI > 0)[0][0] - 1
                            n = stack[2*iPos+1]
                            m = stack[2*iPos+2]

                            stack[2*iPos+2] = -m - self.bix2[iPos] * n

                        elif XI[0] > 0:
                            # If X0>0, flip  mhat^i corresponding to one of the X^i=0
                            iZero = np.where(XI == 0)[0][0] - 1
                            n = stack[2*iZero+1]
                            m = stack[2*iZero+2]

                            stack[2*iZero+2] = -m - self.bix2[iZero] * n

                        else:
                            # If X0<0, flip n^i corresponding to one of the X^i=0
                            iZero = np.where(XI == 0)[0][0] - 1
                            n = stack[2*iZero+1]
                            m = stack[2*iZero+2]

                            stack[2*iZero+1] = -n
                            stack[2*iZero+2] = m + self.bix2[iZero] * n

                    else:
                        # Type A', the rest of type B', and type C': simply flip all signs
                        stack[1:] *= -1

                    if np.random.rand() < 0.5:
                        # There is no natural way to fix the overall sign of the YI during
                        # the above transformations. Set at random by flipping all mhat signs.
                        for i in range(3):
                            n = stack[2*i+1]
                            m = stack[2*i+2]

                            stack[2*i+2] = -m - self.bix2[i] * n

        # Remove all type C branes
        if env == 3:

            YaI = [getStackXY(stack, self.bix2)[1] for stack in self.chromosome]
            typeC = np.where([YI @ YI == 0 for YI in YaI])[0]

            self.chromosome = np.delete(self.chromosome, typeC, axis=0)

    # Bring chromosome to standard form
    def standardize(self):
        # For each stack, use the ambiguity in representing
        # brane stacks with winding numbers to pick a 'standard form'.
        # Flipping the signs of two (n,m) pairs simultaneously does
        # not change the XI, YI. Use this to choose the representation
        # that is lexicographically last.

        for stack in self.chromosome:

            # Get ns and ms
            n1, m1, n2, m2, n3, m3 = stack[1:]

            # Flip pairs 1 and 3
            if n1 < 0 or (n1 == 0 and m1 < 0):
                stack[1:3] *= -1
                stack[5:7] *= -1

            # Flip pairs 2 and 3
            if n2 < 0 or (n2 == 0 and m2 < 0):
                stack[3:7] *= -1

        # Combine all pairs of stacks with indentical XI, YI
        pairs = getIdenticalPairs(self.chromosome, self.bix2)

        while len(pairs) > 0:
            # Select one pair at random
            pair = pairs[np.random.randint(len(pairs))]
            a, b = np.random.permutation(pair)

            # Absorb Nb into Na and delete stack b
            self.chromosome[a, 0] += self.chromosome[b, 0]
            self.chromosome = np.delete(self.chromosome, b, axis=0)

            pairs = getIdenticalPairs(self.chromosome, self.bix2)

    # Compute the individual's fitness given environment, complex structure moduli and weights
    def updateFitness(self, env, UI, weights, tadScale, susyScale):

        # Extract and record Na and XaI/YaI
        Na = self.chromosome[:, 0]
        XYaI = np.array([getStackXY(stack, self.bix2) for stack in self.chromosome])
        XaI = XYaI[:, 0]
        YaI = XYaI[:, 1]
        self.XaI = XaI
        self.YaI = YaI

        # Tadpole condition: these four integers should be zero
        tadpoles = np.dot(Na, XaI) - 8

        # Number of type C branes of each type
        self.numTypeC = [0, 0, 0, 0]

        # For env=3, type C (filler) branes are added by hand when evaluating the fitness
        if env == 3:

            # Type C branes contribute (1+2b1)(1+2b2)(1+2b3) to X0 tadpole
            if tadpoles[0] < 0:
                self.numTypeC[0] = np.floor(-tadpoles[0] / np.product(1+self.bix2))
                tadpoles[0] += self.numTypeC[0] * np.product(1+self.bix2)

            # Type C branes contribute 1+2bi to Xi tadpoles
            for i in range(3):
                if tadpoles[i+1] < 0:
                    self.numTypeC[i+1] = np.floor(-tadpoles[i+1] / (1+self.bix2[i]))
                    tadpoles[i+1] += self.numTypeC[i+1] * (1+self.bix2[i])

        self.tadpoles = tadpoles

        # K-theory constraint: these four integers should be even
        Kth = np.dot(Na, YaI) % 2
        self.Kth = Kth

        # U0dual = U1*U2*U3, U1dual = U0*U2*U3, etc.
        UIdual = np.array(np.product(UI) / UI, dtype='int')

        # SUSY conditions
        #  - SUSY-X: sum(XI*UI) must be positive for each stack
        #  - SUSY-Y: sum(YI*UIdual) = 0 for each stack
        susyX = np.array([np.minimum(0, XI @ UI) for XI in XaI]) / np.sum(UI)
        susyY = np.array([YI @ UIdual for YI in YaI]) / np.sum(UIdual)

        self.susyXY = abs(susyX) + abs(susyY)

        # Compute and record individual terms in fitness function
        Tfitness = hyperbola(np.mean(abs(tadpoles)) / tadScale)
        Kfitness = np.mean(Kth)
        Sfitness = hyperbola(np.mean(abs(susyX) + abs(susyY)) / susyScale)
        MSSMfitness = self.MSSM()

        self.fitnessTerms = np.array([Tfitness, Kfitness, Sfitness, MSSMfitness])

        # Compute fitness as weighted sum
        self.fitness = weights @ self.fitnessTerms

        # Record which consistency conditions are satisfied
        self.consCond = ''
        if max(abs(tadpoles)) == 0:
            self.consCond += 'T'
        if max(Kth) == 0:
            self.consCond += 'K'
        if max(abs(susyX)) + max(abs(susyY)) == 0:
            self.consCond += 'S'

    # Computes the MSSM fitness
    def MSSM(self):

        Na = self.chromosome[:, 0]

        notTypeC = [(YI != 0).any() for YI in self.YaI]

        U3inds = np.where((Na == 3) * (notTypeC))[0]
        U2inds = np.where((Na == 2) * (notTypeC))[0]
        U1inds = np.where((Na == 1) * (notTypeC))[0]

        # Compute 'distance' to U(3)xU(2)xU(1)xU(1)
        U3dist = max(1-len(U3inds), 0)
        U2dist = max(1-len(U2inds), 0)
        U1dist = max(2-len(U1inds), 0)

        # This is one of 0,1,2,3,4 (smaller better)
        U3211dist = U3dist + U2dist + U1dist

        MSSMfitness = U3211dist/4

        return MSSMfitness

    # Determines brane classification for each stack
    def braneTypes(self):
        # Return array of counts of brane types in the following order:
        #    A, B, C, A', B', C', D', E'

        counts = np.zeros(8)
        types = np.array(["" for i in range(len(self.chromosome))])

        for a in range(len(self.chromosome)):

            XI = self.XaI[a]
            YI = self.YaI[a]

            nZeroXI = np.sum(XI == 0)
            nZeroYI = np.sum(YI == 0)
            nPosXI = np.sum(XI > 0)

            if nZeroYI == 0:
                if nPosXI == 3:
                    # Type A
                    types[a] = "A"
                    counts[0] += 1
                else:
                    # Type A'
                    types[a] = "A'"
                    counts[3] += 1

            elif nZeroXI == 2:
                if nPosXI == 2:
                    # Type B
                    types[a] = "B"
                    counts[1] += 1
                else:
                    # Type B'
                    types[a] = "B'"
                    counts[4] += 1

            elif nZeroXI == 3 and nZeroYI == 4:
                if nPosXI == 1:
                    # Type C
                    types[a] = "C"
                    counts[2] += 1
                else:
                    # Type C'
                    types[a] = "C'"
                    counts[5] += 1

            elif nZeroXI == 3 and nZeroYI == 3:
                # Type D'
                types[a] = "D'"
                counts[6] += 1

            else:
                # Type E'
                types[a] = "E'"
                counts[7] += 1

        return types, counts

    # Sort stacks in chromosome to avoid duplicates when saving
    def saveSort(self):
        # Sort stacks in reverse-lexicagraphical order for saving.
        order = np.lexsort(self.chromosome.T[::-1])[::-1]
        return self.chromosome[order].copy()

    # Display the individual's chromosome and details of fitness
    def display(self):

        print(' Na |  n1  m1  n2  m2  n3  m3   |   X0  X1  X2  X3   |   Y0  Y1  Y2  Y3\n' + '-'*72)

        for i in range(len(self.chromosome)):
            stack = self.chromosome[i]
            print('{:3d} |'.format(stack[0]), end='')
            for s in stack[1:]:
                print('{:4d}'.format(s), end='')
            print('   | ', end='')
            for XI in self.XaI[i]:
                if XI == 0:
                    print(' '*4, end='')
                else:
                    print('{:4d}'.format(XI), end='')
            print('   | ', end='')
            for YI in self.YaI[i]:
                if YI == 0:
                    print(' '*4, end='')
                else:
                    print('{:4d}'.format(YI), end='')
            print('')
        print(' '*34 + '-'*38 + '\n' + ' '*34, end='')
        for tad in self.tadpoles:
            print('%4d' % tad, end='')
        print('   | ', end='')
        for k in self.Kth:
            print('%4d' % k, end='')
        print('\nFitness: %.4f\n' % self.fitness)


# Returns phenomological properties of a chromosome
def getInfo(chromosome, env, bix2, UI, maxSUN):

    k = len(chromosome)

    # Set up individual
    ind = individual(bix2, k-1, k+1)
    ind.chromosome = chromosome
    ind.updateFitness(env, UI, np.array([0, 0, 0, 0]), 1, 1)

    # Extract data
    Tmean = np.mean(abs(ind.tadpoles))
    Kmean = np.mean(ind.Kth)
    Smean = np.mean(ind.susyXY)

    # print(ind.XaI, ind.YaI)

    rank = sum(chromosome[:, 0]) + sum(ind.numTypeC)
    rank = int(rank)

    types, counts = ind.braneTypes()

    ABCcounts = np.zeros(3, dtype='int')
    ABCcounts[0] = np.sum(types == "A")
    ABCcounts[1] = np.sum(types == "B")
    ABCcounts[2] = np.sum(types == "C")

    ABCcounts[2] += sum(ind.numTypeC)

    # Get numbers of SU(N) factors
    SUNcounts = np.zeros(maxSUN + 1, dtype='int')
    for a in range(len(chromosome)):
        if types[a] != "C" and types[a] != "C'":
            Na = chromosome[a, 0]
            if Na <= maxSUN:
                SUNcounts[Na] += 1

    meanChirality = 0
    counter = 0
    for a in range(len(ind.chromosome)-1):
        for b in range(a+1, len(ind.chromosome)):
            XaI = ind.XaI[a]
            YbI = ind.YaI[b]

            counter += 1

            meanChirality += 2*(XaI@YbI)

    meanChirality /= (1+bix2[0]) * (1+bix2[1]) * (1+bix2[2])
    meanChirality /= k*(k-1)/2
    meanChirality = int(meanChirality)

    Qbest = np.zeros(3, dtype='int')
    Lbest = np.zeros(2, dtype='int')
    U1masslessbest = 0
    alphasbest = np.zeros(3)

    if (SUNcounts[1] >= 2) and (SUNcounts[2] >= 1) and (SUNcounts[3] >= 1):
        U1inds = np.where((chromosome[:, 0] == 1) * ((types == 'A') + (types == 'B')))[0]
        U2inds = np.where((chromosome[:, 0] == 2) * ((types == 'A') + (types == 'B')))[0]
        U3inds = np.where((chromosome[:, 0] == 3) * ((types == 'A') + (types == 'B')))[0]

        distBest = np.inf

        for U3a in U3inds:
            for U2b in U2inds:
                for U1c in U1inds:
                    for U1d in U1inds:
                        if U1d == U1c:
                            continue

                        QL = getIab(ind.chromosome[U3a], ind.chromosome[U2b],
                                    ind.bix2, False, False)
                        QL += getIab(ind.chromosome[U3a], ind.chromosome[U2b],
                                     ind.bix2, False, True)

                        uR = getIab(ind.chromosome[U3a], ind.chromosome[U1c],
                                    ind.bix2, True, False)
                        uR += getIab(ind.chromosome[U3a], ind.chromosome[U1d],
                                     ind.bix2, True, False)

                        dR = getIab(ind.chromosome[U3a], ind.chromosome[U1c],
                                    ind.bix2, True, True)
                        dR += getIab(ind.chromosome[U3a], ind.chromosome[U1d],
                                     ind.bix2, True, True)
                        dR += (getIab(ind.chromosome[U3a], ind.chromosome[U3a],
                                      ind.bix2, False, True)
                               + getIaO6(ind.chromosome[U3a], ind.bix2, False)) / 2
                        dR = int(dR)

                        L = getIab(ind.chromosome[U2b], ind.chromosome[U1c],
                                   ind.bix2, False, False)
                        L += getIab(ind.chromosome[U2b], ind.chromosome[U1d],
                                    ind.bix2, False, False)
                        L += getIab(ind.chromosome[U2b], ind.chromosome[U1c],
                                    ind.bix2, True, False)
                        L += getIab(ind.chromosome[U2b], ind.chromosome[U1d],
                                    ind.bix2, True, False)

                        eR = (getIab(ind.chromosome[U1c], ind.chromosome[U1c],
                                     ind.bix2, False, True)
                              - getIaO6(ind.chromosome[U1c], ind.bix2, False)) / 2
                        eR += (getIab(ind.chromosome[U1d], ind.chromosome[U1d],
                                      ind.bix2, False, True)
                               - getIaO6(ind.chromosome[U1d], ind.bix2, False)) / 2
                        eR += getIab(ind.chromosome[U1c], ind.chromosome[U1d],
                                     ind.bix2, False, True)
                        eR = int(eR)

                        U1massless = int((ind.YaI[U3a] + ind.YaI[U1c] + ind.YaI[U1d] == 0).all())

                        alpha_a = (ind.XaI[U3a] @ UI)**(-1.0)
                        alpha_b = (ind.XaI[U2b] @ UI)**(-1.0)
                        alpha_c = (ind.XaI[U1c] @ UI)**(-1.0)
                        alpha_d = (ind.XaI[U1d] @ UI)**(-1.0)

                        alpha_Y = (1/(6*alpha_a) + 1/(2*alpha_c) + 1/(2*alpha_d))**(-1)

                        dist = np.var([QL, uR, dR]) + np.var([L, eR])

                        if dist < distBest:
                            distBest = dist
                            Qbest = np.asarray([QL, uR, dR], dtype='int')
                            Lbest = np.asarray([L, eR], dtype='int')
                            U1masslessbest = U1massless
                            alphasbest = np.array([alpha_a, alpha_b, alpha_Y])

    return np.array([Tmean, Kmean, Smean, rank, meanChirality, ABCcounts, SUNcounts,
                     Qbest, Lbest, U1masslessbest, alphasbest])


# Returns XI/YI for a single stack
def getStackXY(stack, bix2):

    n1, m1, n2, m2, n3, m3 = stack[1:]

    m1hat = m1 + bix2[0]*(n1 + m1)
    m2hat = m2 + bix2[1]*(n2 + m2)
    m3hat = m3 + bix2[2]*(n3 + m3)

    X0 = + n1    * n2    * n3
    X1 = - n1    * m2hat * m3hat
    X2 = - m1hat * n2    * m3hat
    X3 = - m1hat * m2hat * n3

    Y0 = + m1hat * m2hat * m3hat
    Y1 = - m1hat * n2    * n3
    Y2 = - n1    * m2hat * n3
    Y3 = - n1    * n2    * m3hat

    XI = np.array([X0, X1, X2, X3])
    YI = np.array([Y0, Y1, Y2, Y3])

    return [XI, YI]


# Compute intersection number between two stacks or their orientifold image(s)
def getIab(stack1, stack2, bix2, orientImage1, orientImage2):
    # orientImage1/2 are booleans to indicate orientifold image (Y -> -Y)

    X1, Y1 = getStackXY(stack1, bix2)
    X2, Y2 = getStackXY(stack2, bix2)

    if orientImage1:
        Y1 *= -1
    if orientImage2:
        Y2 *= -1

    Iab = np.product(1-bix2/2) * (X1@Y2 - Y1@X2)

    return int(Iab)


# Compute intersection number between a stack and the O6 planes
def getIaO6(stack, bix2, orientImage):
    # orientImage is a boolean to indicate orientifold image (Y -> -Y)

    Xa, Ya = getStackXY(stack, bix2)
    XO6 = 4 * np.ones(4)
    YO6 = np.zeros(4)

    if orientImage:
        Ya *= -1

    Iab = np.product(1-bix2/2) * (Xa@YO6 - Ya@XO6)

    return int(Iab)


# Find all pairs of stacks in a chromosome which wrap the same homological cycle
def getIdenticalPairs(chromosome, bix2):
    # Two stacks wrap the same homological cycle iff their XaI/YaI are the same

    pairs = np.empty([0, 2], dtype='int')

    for a in range(len(chromosome)):
        XaI, YaI = getStackXY(chromosome[a], bix2)

        for b in range(a+1, len(chromosome)):
            XbI, YbI = getStackXY(chromosome[b], bix2)

            if (XaI == XbI).all() and (YaI == YbI).all():
                pairs = np.append(pairs, [[a, b]], axis=0)

    return pairs


# Returns random winding numbers and multiplicities to initialize a single stack
def randomStack(p, mu):

    # Random Na and winding numbers
    stack = np.zeros(7, dtype='int')
    stack[0] = np.random.geometric(p)
    stack[1:] = randomSkellam(mu, 6)

    return stack


# Returns n Skellam random variables
def randomSkellam(mu, n):
    # If Y,Z~Pois(mu), then X=Y-Z follows the distribution X~Skellam(mu,mu).
    # This discrete probability distribution has support
    # on all integers, mean zero and variance 2*mu.
    pois1 = np.random.poisson(mu, size=n)
    pois2 = np.random.poisson(mu, size=n)

    return pois1 - pois2


# Hyperbola passing through (0,0) and asymptoting to 1 for z -> inf
def hyperbola(z):
    return z / (1 + z)
