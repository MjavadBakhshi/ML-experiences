import random
import numpy as np


# create a permutation of 8 queen place.
# output is a vector with size 8 that each element of it
# represent row place of queen and index of element refer
# to i th queen
# example 2, 5,1,4,6,7,6,8
# this vector means that first queen placed in column 1 and row 2
# second queen placed in column 2 and row 5,...
def random_individual(size):
    return [random.randint(1, 8) for _ in range(8)]


# The ideal case can yield upton 28 arrangements of non attacking pairs.
maxFitness = 28


# calculate fitness of each individuals.
def fitness(individual):

    # calculate row and column collisions
    # just subtract the unique length of array from total length of array
    # [1,1,1,2,2,2] - [1,2] => 4 clashes
    horizontal_collisions =  abs(len(individual) - len(np.unique(individual)))
    diagonal_collisions = 0

    # calculate diagonal clashes
    for i in range(len(individual)):
        for j in range(len(individual)):
            if i != j:
                dx = abs(i - j)
                dy = abs(individual[i] - individual[j])
                if dx == dy:
                    diagonal_collisions += 1

    return int(maxFitness - (horizontal_collisions + diagonal_collisions))


def probability(individual):
    return fitness(individual) / maxFitness


# implementation of wheel roulette
def random_pick(population, probabilities):
    total = np.sum(probabilities)
    r = random.uniform(0, total)
    upto = 0
    for _ in range(len(probabilities)):
        if upto + probabilities[_] >= r:
            return population[_]
        upto += probabilities[_]


# cross over of two parent
def reproduce(x, y):
    n = len(x)
    # select cut off point
    c = random.randint(0, n - 1)
    return x[0:c] + y[c:n]


# mutate a gen randomly
# select a queen place randomly and set with a
# queen randomly
# example: 2,2,3,4,4,5,6,8
# select 3th place and change it to 5
# final individual (chromosome) is
# 2,2,5,4,4,5,6,8
def mutate(x):
    n = len(x)
    c = random.randint(0, n - 1)
    m = random.randint(1, n)
    x[c] = m
    return x


# Base GA :
# do selection, crossover, mutation, and create population for
# new generation.
def genetic_queen(population):
    mutation_probability = 0.03
    # --create new population
    new_population = []
    # calculate probability of each individual.
    probabilities = [probability(n) for n in population]
    # --selection phase.
    # select parent with wheel roulette method.
    for i in range(len(population)):
        x = random_pick(population, probabilities)
        y = random_pick(population, probabilities)
        # --cross over phase.
        # create new child by cross over.
        child = reproduce(x, y)
        # -- mutation phase.
        # mutate with mutation_probability
        # for this create a random probability
        # and compare with mutation  probability
        if random.random() < mutation_probability:
            child = mutate(child)

        print_individual(child)
        new_population.append(child)
        # check find solution or no?
        if fitness(child) == 28:
            break
    return new_population


def print_individual(x):
    print("{},  fitness = {}, probability = {:.6f}"
          .format(str(x), fitness(x), probability(x)))



# -- main program:

# create initial population
population = [random_individual(8) for _ in range(100)]
print("some initial individuals :")
print(population[0:10])


# set generation to 1.
generation = 1

# continue until find solution.
# [fitness(x) for x in population] this part , first run and create a array
# that contains fitness of each individual.
# then this part of code will be run while not 28 in
# if a solution has  fitness equal 28, that means solution has been find.
while not 28 in [fitness(x) for x in population]:
    print("=== Generation {} ===".format(generation))
    population = genetic_queen(population)

    print("Maximum fitness = {}".format(max([fitness(n) for n in population])))
    generation += 1

print("Solved in Generation {}!".format(generation - 1))
for x in population:
    if fitness(x) == 28:
        print_individual(x)
