from numpy.random import randint
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as pyplot

Tfidf = np.array(np.loadtxt("E:\\nnProject2022\Tf-idf_tables\\tf-idf.csv", delimiter=",", dtype=np.float16))

# Genetic Representation of a Solution (Initial Population)
def init_population(numOfChromosomes, prob):  
    initial_population=np.random.choice([0, 1], size=(numOfChromosomes,8520), p=[1-prob, prob])
    initial_population=initial_population.tolist()
    return initial_population


# Repair function
def repair_population(population, numOfChromosomes): #pop_array = 2-d array | pop = number of genomes in population
    for i in range(0, numOfChromosomes):
        while(sum(population[i])<1000):
            cnt=sum(population[i])
            replace_indx = np.random.randint(0, 8520, 1000-cnt)# fill in aces
            population[i,replace_indx]=1
    return population


# Fitness Function
def fitness_function(population, numOfChromosomes):
    fitness_scores=np.zeros(numOfChromosomes)
    for i in range(0, numOfChromosomes):  #for each chromosome
        tfidf=0
        counter=0
        for j in range(0, 8520): #for each gene
            if(population[i][j]==1): #if value is 1
                tfidf += Tfidf[j] # add it's tfidf value to a variable
                counter += 1 # count how many values are added to calc average
        fitness_scores[i]=(tfidf/counter)-((counter-1000)/8520*(tfidf/counter)) #calculate the average tfidf value and subtract the >1000 words penalty
    fitness_scores=fitness_scores.tolist()
    return fitness_scores

 
# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # pt1 = randint(1, len(p1)-2)
        # pt2 = randint(pt1, len(p2)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        # c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
        # c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
    return [c1, c2]
 
# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
 
# genetic algorithm
def genetic_algorithm(n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = init_population(n_pop,0.2)
    # keep track of best solution
    best, best_eval = 0, 0
    #score table
    score_list=[]
    gen_list=[]
    history=[]
    # enumerate generations
    for gen in range(n_iter):
        # repair population
        pop=repair_population(pop,n_pop)
        # evaluate all candidates in the population
        scores = fitness_function(pop,n_pop)
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best = %.7f" % (gen, scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        #fill lists for plot
        gen_list.append(gen)
        score_list.append(best_eval)


        # if(gen>1 and score_list[gen]-score_list[gen-1]<1/1000000*score_list[gen]):
        #     print('Not significant improvement (<0.00001%)')
        #     print(gen)
        #     break

        if(gen>20 and score_list[gen]==score_list[gen-20]):  #and score_list[gen]-score_list[gen-1]>1/1000000*score_list[gen]
            print('Steady for 20 generations')
            print(gen)
            break


    return [best, best_eval, score_list, gen_list]
 
# define the total iterations
n_iter = 100
# define the population size
n_pop = 200
# crossover rate
r_cross = 0.1
# mutation rate
r_mut = 0.01 #1.0/8520
# perform the genetic algorithm search
best, best_eval, score_list, gen_list = genetic_algorithm(n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('score = %f' % (best_eval))
pyplot.plot(gen_list,score_list)
pyplot.show()