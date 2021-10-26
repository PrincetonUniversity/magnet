materials = {
    # material => (k_i, alpha, beta)
    'N27': (4.88e-10, 1.09, 2.44),
    'N87': (5.77e-12, 1.43, 2.49),
    'N49': (1.18e-12, 1.27, 3.17)
}

material_names = list(materials.keys())
excitations = ('Datasheet', 'Sinusoidal', 'Triangle', 'Trapezoidal')
prediction_algorithms = ('iGSE', 'Advanced Analytical', 'Machine Learning')
