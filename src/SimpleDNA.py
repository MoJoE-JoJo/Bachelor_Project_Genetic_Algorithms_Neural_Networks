from random import random
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer
from src.Genes.DenseGene import DenseGene
from src.Genes.OverallGene import OverallGene


# Contains two genes, one overall gene and one dense gene.
class SimpleDNA:
    def __init__(self):
        no_dense_genes = 1
        max_no_nodes = 1000

        overall_gene = [OverallGene(Optimizer(random.randrange(1, len(Optimizer)+1)),
                                    Loss(random.randrange(1, len(Loss)+1)))]
        dense_genes = [DenseGene(random.randrange(0, max_no_nodes+1),
                                 Activation(random.randrange(1, len(Activation)+1)),
                                 max_no_nodes)
                       for i in range(no_dense_genes)]

        self.genes = overall_gene + dense_genes
