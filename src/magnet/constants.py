materials = {
    # material => (k_i, alpha, beta)
    'N27': (3.72e-04, 1.52, 2.53),
    'N87': (9.80e-04, 1.47, 2.61),
    'N49': (1.69e-03, 1.50, 3.23),
    '3C90': (2.54e-04, 1.54, 2.62),
    '3C94': (5.90e-05, 1.62, 2.50),
    '3E6': (2.8e-07, 1.91, 2.09),
    '3F4': (9.39e-03, 1.41, 3.15),
    '77': (2.92e-04, 1.53, 2.52),
    '78': (8.77e-05, 1.61, 2.54),
    'N30': (3.3e-07, 1.96, 2.35),
}

material_names = list(materials.keys())
excitations = ('Datasheet', 'Sinusoidal', 'Triangular', 'Trapezoidal')
prediction_algorithms = ('iGSE', 'Advanced Analytical', 'Machine Learning')
