import numpy as np

nps_classes = [
    'Cathinone analogues',
    'Cannabinoid analogues',
    'Phenethylamine analogues',
    'Piperazine analogues',
    'Tryptamine analogues',
    'Fentanyl analogues'
]

def random_position(free_cells):
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(y), float(x)
