import numpy as np

def create_genotype_matrix():
    SNP_features = np.random.randint(10, 101)
    samples = np.random.randint(1000, 5001)

    genotype_matrix = np.empty((SNP_features, samples))

    for i, row in enumerate(genotype_matrix):
        p = np.random.random()
        for j, col in enumerate(row):
            genotype_matrix[i][j] = np.random.binomial(2, p)

    return genotype_matrix


if __name__ == "__main__":
    genotype_matrix = create_genotype_matrix()
    print(genotype_matrix)
