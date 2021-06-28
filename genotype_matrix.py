import numpy as np

def create_genotype_matrix():
    snps = np.random.randint(10, 101)
    genes = np.random.randint(1000, 5001)

    genotype_matrix = np.empty((snps, genes))

    for i, row in enumerate(genotype_matrix):
        for j, col in enumerate(row):
            p = np.random.random()
            genotype_matrix[i][j] = np.random.binomial(2, p)

    return genotype_matrix


if __name__ == "__main__":
    genotype_matrix = create_genotype_matrix()
    print(genotype_matrix)
