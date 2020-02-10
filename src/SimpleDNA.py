from random import random
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer
from src.Genes.DenseGene import DenseGene
from src.Genes.OverallGene import OverallGene


# Contains two genes, one overall gene and one dense gene.
class SimpleDNA:
    fitness = 0.0
    mutation_rates = [0.2, 0.6, 0.2]    #Represents individual probabilities of mutating either 0, 1, or 2 genes

    def __init__(self):
        no_dense_genes = 1
        initial_max_no_nodes = 1000

        overall_gene = [OverallGene(Optimizer(random.randrange(1, len(Optimizer)+1)),
                                    Loss(random.randrange(1, len(Loss)+1)))]
        dense_genes = [DenseGene(random.randrange(0, initial_max_no_nodes+1),
                                 Activation(random.randrange(1, len(Activation)+1)))
                       for i in range(no_dense_genes)]

        self.genes = overall_gene + dense_genes

    # uses the normalized mutations rates as probabilities for the number of mutations
    def mutate(self):
        n_mutation_rates = self.__normalize()
        mutation = random.uniform(0.0, 1.0)
        if mutation <= n_mutation_rates[0]:
            return
        elif mutation <= n_mutation_rates[0] + n_mutation_rates[1]:
            self.genes[random.randrange(0, len(self.genes))].mutate()
        elif mutation <= sum(n_mutation_rates):
            self.genes[0].mutate()
            self.genes[1].mutate()

    # returns normalized versions of the different mutation rates
    def __normalize(self):
        m_rate_sum = sum(self.mutation_rates)
        n_mutation_rates = [i/m_rate_sum for i in self.mutation_rates]
        return n_mutation_rates


    #fitness function, modtager input-shape, output shape, og datasættet, generer det neurale netværk, og køre det. Evaluerer det med cross evaluation
