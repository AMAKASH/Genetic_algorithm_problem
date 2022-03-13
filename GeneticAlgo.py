import numpy as np


class GeneticAlgorithmStrangeBankProblem:
    start_population_size = 10
    mutation_threshold = 0.3

    def __init__(self, path):
        self.itr_limit = 100
        self.n = -1
        self.fittingData = []
        self.population = []
        file = open(path, 'r')
        data = file.readlines()
        # print(data)
        file.close()
        for i, d in enumerate(data):
            temp = d[:-1].split()
            if i == 0:
                self.n = int(temp[0])
            elif temp[0] == 'l':
                self.fittingData.append(int(temp[1]) * -1)
            elif temp[0] == 'd':
                self.fittingData.append(int(temp[1]))

        print(self.n)
        print(self.fittingData)
        self.best_fit_population = self.calc_goal_state()

    def get_best_fit(self):
        if np.any(self.best_fit_population == -1):
            return "-1"
        string_rep = np.array2string(self.best_fit_population, separator="")
        return string_rep[1:-1]

    def calc_goal_state(self):
        self.population = np.random.randint(0, 2, (GeneticAlgorithmStrangeBankProblem.start_population_size, self.n))
        for i in range(self.itr_limit):
            print("GEN:", i + 1)
            new_population = []
            fitness_scores = self.fitness()
            max_index = np.where(fitness_scores == max(fitness_scores))[0]
            if 0 in fitness_scores:
                index = np.where(fitness_scores == 0)[0][0]
                print("Best Fit Candidate Index:", index)
                print("Best fit Candidate:", self.population[index])
                if np.all(self.population[index] == 0):
                    self.population[index] = self.population[max_index]
                    print("Not Acceptable Candidate")
                else:
                    return self.population[index]
            for j in range(GeneticAlgorithmStrangeBankProblem.start_population_size):
                x, y = self.select(fitness_scores)
                child = self.crossover(x, y)
                child = self.mutate(child)
                if np.all(child == 0):
                    continue
                new_population.append(child)
            self.population = new_population
        return np.array([-1])

    def fitness(self):
        print("Population: ", self.population)
        score = []
        for people in self.population:
            total = 0
            for i in range(len(people)):
                if people[i] == 1:
                    total += self.fittingData[i]
            score.append(total)
        score = np.absolute(np.array(score))
        print("Fit Score:", score)
        return score

    def select(self, scores):
        total = np.sum(scores)
        # print("Total", total)
        prob = total - scores
        prob = prob / np.sum(prob)
        print("Probability:", prob)
        # print(np.sum(prob))
        indices = []
        for i in range(len(self.population)):
            indices.append(i)
        choice = np.random.choice(indices, 2, True, prob)
        print(choice)
        return self.population[choice[0]], self.population[choice[1]]

    def crossover(self, x, y):

        crossover_index = np.random.randint(0, self.n)
        child = np.append(x[:crossover_index], y[crossover_index:])
        print("X", x)
        print("Y", y)
        print("Cross-over INdex:", crossover_index)
        print("Child: ", child)
        return child

    def mutate(self, child):

        prob = GeneticAlgorithmStrangeBankProblem.mutation_threshold
        if np.random.random(1)[0] > prob:
            mutation_index = np.random.randint(0, self.n)
            mutation_val = np.random.randint(0, 2)
            print("Before Mutation:", child, mutation_index, mutation_val)
            child[mutation_index] = mutation_val
            print("After Mutation:", child)
        return child


def runner():
    # Input 1
    bank_records1 = GeneticAlgorithmStrangeBankProblem('input1.txt')
    print("OUTPUT 1:", bank_records1.get_best_fit())

    # Input 2
    # bank_records2 = GeneticAlgorithmStrangeBankProblem('input2.txt')
    # print("OUTPUT 2:", bank_records2.get_best_fit())


if __name__ == '__main__':
    runner()
