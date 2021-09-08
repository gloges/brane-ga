import numpy as np


class individual:

    def __init__(self, bix2, kMin, kMax):

        self.bix2 = np.array(bix2)
        self.kMin = kMin
        self.kMax = kMax

        # Initialize chromosome to contain k stacks
        k = np.random.randint(self.kMin, self.kMax + 1)
        self.chromosome = np.array([randomStack(1/3, 10) for i in range(k)])


        self.fitness = None


    def crossOver(self, parent2, xoverProbs):
        # xoverProbs is an array of four probabilities
        # for the four cross-over methods.

        # Randomly choose a cross-over method
        rnd = np.random.rand()
        cumulative = np.cumsum(xoverProbs)
        method = sum(cumulative < rnd)


        # All four methods use uniform, 1-point cross-overs
        # that ensure the child has between 2 and kMax stacks.

        if method == 0:
            # Cross-over point is between stacks (i.e. stacks are indivisible).
            # 'rows1' rows are taken from the beginning of parent1 (self) and
            # 'rows2' rows are taken from the end of parent2.
            rows1 = np.inf
            rows2 = np.inf

            while rows1 + rows2 > self.kMax or rows1 + rows2 < self.kMin:
                rows1 = np.random.randint(len(self.chromosome)) + 1
                rows2 = np.random.randint(len(parent2.chromosome)) + 1

            newChromosome = np.zeros([rows1 + rows2, 7], dtype='int')
            newChromosome[:rows1] = self.chromosome[:rows1]
            newChromosome[-rows2:] = parent2.chromosome[-rows2:]

        elif method == 1:
            # Cross-over point is between stacks (i.e. stacks are indivisible)
            # and the child has the same number of stacks as parent1 (self)
            newChromosome = self.chromosome.copy()
            rows2 = np.random.randint(1, min(len(parent2.chromosome)+1, len(self.chromosome)))

            # Either overwrite rows from beginning of parent2 on beginning of parent1
            # or rows from end of parent2 on end of parent1
            if np.random.rand() < 0.5:
                newChromosome[:rows2] = parent2.chromosome[:rows2]
            else:
                newChromosome[-rows2:] = parent2.chromosome[-rows2:]

        elif method == 2:
            # Cross-over point is between genes (i.e. stacks are divisible)

            # rows1/rows2 are the number of *full* rows from each parent
            # col is the number of genes from parent1 in the split row
            rows1 = np.inf
            rows2 = np.inf
            col = np.random.randint(1, 7)

            while (rows1 + rows2 + 1 > self.kMax) or (rows1 + rows2 + 1 < self.kMin):
                rows1 = np.random.randint(len(self.chromosome))
                rows2 = np.random.randint(len(parent2.chromosome))

            newChromosome = np.zeros([rows1 + rows2 + 1, 7], dtype='int')

            # Full rows from parent1
            newChromosome[:rows1] = self.chromosome[:rows1]

            # Full rows (if any) from parent2
            if rows2 > 0:
                newChromosome[-rows2:] = parent2.chromosome[-rows2:]

            # Split row
            newChromosome[rows1, :col] = self.chromosome[rows1, :col]
            newChromosome[rows1, col:] = parent2.chromosome[-(rows2+1), col:]

        elif method == 3:
            # Cross-over point is between genes (i.e. stacks are divisible)
            # and the child has the same number of stacks as parent1 (self)

            newChromosome = self.chromosome.copy()
            rows2 = np.random.randint(min(len(parent2.chromosome), len(self.chromosome)))
            col = np.random.randint(1, 7)

            # Either overwrite genes from beginning of parent2 on beginning of parent1
            # or genes from end of parent2 on end of parent1
            if np.random.rand() < 0.5:
                newChromosome[:rows2] = parent2.chromosome[:rows2]
                newChromosome[rows2, :col] = parent2.chromosome[rows2, :col]
            else:
                newChromosome[-(rows2+1):] = parent2.chromosome[-(rows2+1):]
                newChromosome[-(rows2+1), :col] = self.chromosome[-(rows2+1), :col]

        else:
            print('Error: sum of cross-over probabilities is < 1')


        child = individual(self.bix2, self.kMin, self.kMax)
        child.chromosome = newChromosome

        return child


    def mutate(self, mutRates):
        
        mutStackSplit, mutStackPerm = mutRates[:2]

        mutNaPM, mutWindPM, mutWindSgns, mutS4 = mutRates[2:]



        # Split stack with Na > 1 into two
        if len(self.chromosome) < self.kMax and np.random.rand() < mutStackSplit:

            splittable = np.where(self.chromosome[:, 0] > 1)[0]
            if len(splittable) > 0:
                a = np.random.choice(splittable)
                b = np.random.randint(len(self.chromosome) + 1)
                Na = self.chromosome[a, 0]
                self.chromosome = np.insert(self.chromosome, b, self.chromosome[a], axis=0)
                self.chromosome[a, 0] = np.random.randint(1, Na)
                self.chromosome[b, 0] = Na - self.chromosome[a, 0]


        # Mutations to each stack
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
                        stack[2*i+2] = m + self.bix2[i] * n
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


        # Permute stacks
        if np.random.rand() < mutStackPerm:
            np.random.shuffle(self.chromosome)




    def adjust(self, env):

        # The value of env = 1,2,3 gives which types of branes are allowed
        #  - env = 2,3 replace types A',B',C' branes with types A,B,C
        #  - env = 3 then removes all type C branes (but then adds
        #          them by hand for evaluating the fitness function)


        # Correct non-coprime winding number pairs.
        for stack in self.chromosome:

            # Make sure 1 <= Na <= 10
            stack[0] = max(abs(stack[0]), 1)
            stack[0] = min(stack[0], 10)

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
                        # Type A', the rest of type B' and type C': simply flip all signs
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



    def updateFitness(self, env, UI, fitnessParams):

        weights, tadScale, susyScale = fitnessParams
        

        Na = self.chromosome[:, 0]

        XYaI = np.array([getStackXY(stack, self.bix2) for stack in self.chromosome])
        XaI = XYaI[:, 0]
        YaI = XYaI[:, 1]


        # Tadpole condition: these four integers should be zero
        tadpoles = np.dot(Na, XaI) - 8


        # For env=3, type C (filler) branes are added by hand when evaluating the fitness
        if env == 3:
            # Type C branes contribute (1+2b1)(1+2b2)(1+2b3) to X0 tadpole
            if tadpoles[0] < 0:
                tadpoles[0] = -(-tadpoles[0] % np.product(1+self.bix2))

            # Type C branes contribute 1+2bi to Xi tadpoles
            for i in range(3):
                if tadpoles[i+1] < 0:
                    tadpoles[i+1] = -(-tadpoles[i+1] % (1+self.bix2[i]))


        # K-theory constraint: these four integers should be even
        Kth = np.dot(Na, YaI) % 2


        # U0dual = U1*U2*U3, U1dual = U0*U2*U3, etc.
        UIdual = np.array(np.product(UI) / UI, dtype='int')

        # SUSY conditions
        #  - SUSY-X: sum(XI*UI) must be positive for each stack
        #  - SUSY-Y: sum(YI*UIdual) = 0 for each stack
        susyX = np.array([np.minimum(0, XI @ UI) for XI in XaI]) / np.sum(UI)
        susyY = np.array([YI @ UIdual for YI in YaI]) / np.sum(UIdual)


        # Rewards: each lies in the interval [0,1]
        tadNormL1 = sum(abs(tadpoles))

        Treward = (1 + tadNormL1 / tadScale) ** (-1)
        Kreward = np.sqrt(1 - np.mean(Kth))
        Sreward = np.min((1 + (abs(susyX) + abs(susyY)) / susyScale) ** (-1))
        # MSSMreward = MSSM()
        MSSMreward = 0

        self.rewards = np.array([Treward, Kreward, Sreward, MSSMreward])

        self.fitness = weights @ self.rewards


        # Record which consistency conditions are satisfied
        self.consCond = ''
        if max(abs(tadpoles)) == 0:
            self.consCond += 'T'
        if max(Kth) == 0:
            self.consCond += 'K'
        if max(abs(susyX)) + max(abs(susyY)) == 0:
            self.consCond += 'S'


        self.XaI = XaI
        self.YaI = YaI

        self.tadpoles = tadpoles
        self.Kth = Kth


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


    def saveSort(self):
        # Sort stacks in reverse-lexicagraphical order for saving.
        order = np.lexsort(self.chromosome.T[::-1])[::-1]
        return self.chromosome[order]


    def display(self):
        
        if self.fitness is None:
            self.updateFitness()


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



def getInfo(chromosome, env, bix2, UI, fitnessParams, maxSUN):

    k = len(chromosome)

    # Set up individual
    ind = individual(bix2, k-1, k+1)
    ind.chromosome = chromosome
    ind.updateFitness(env, UI, fitnessParams)

    # Extract data
    tadNormL1 = sum(abs(ind.tadpoles))
    KthSum = sum(ind.Kth)

    rank = sum(chromosome[:, 0])

    types, counts = ind.braneTypes()

    ABCcounts = np.zeros(3)
    ABCcounts[0] = np.sum(types == "A")
    ABCcounts[1] = np.sum(types == "B")
    ABCcounts[2] = np.sum(types == "C")

    # Get numbers of SU(N) factors
    SUNcounts = np.zeros(maxSUN + 1)
    for a in range(len(chromosome)):
        if types[a] != "C" and types[a] != "C'":
            Na = chromosome[a, 0]
            if Na <= maxSUN:
                SUNcounts[Na] += 1


    return [tadNormL1, KthSum, rank, ABCcounts, SUNcounts]




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


def getIdenticalPairs(chromosome, bix2):
    # Find all pairs with the same XI, YI

    pairs = np.empty([0, 2], dtype='int')

    for a in range(len(chromosome)):
        XaI, YaI = getStackXY(chromosome[a], bix2)

        for b in range(a+1, len(chromosome)):
            XbI, YbI = getStackXY(chromosome[b], bix2)

            if (XaI == XbI).all() and (YaI == YbI).all():
                pairs = np.append(pairs, [[a, b]], axis=0)

    return pairs


def randomStack(p, mu):

    # Random Na and winding numbers
    stack = np.zeros(7, dtype='int')
    stack[0] = np.random.geometric(p)
    stack[1:] = randomSkellam(mu, 6)

    return stack


def randomSkellam(mu, n):
    # If Y,Z~Pois(mu), then X=Y-Z follows the distribution X~Skellam(mu,mu).
    # This discrete probability distribution has support
    # on all integers, mean zero and variance 2*mu.
    pois1 = np.random.poisson(mu, size=n)
    pois2 = np.random.poisson(mu, size=n)

    return pois1 - pois2
