materials = {
    # material => (k_i, alpha, beta)
    'N27': (0.066924, 1.5158, 2.5254),
    'N87': (0.15178, 1.4722, 2.6147),
    'N49': (0.13263, 1.4987, 3.2337),
    '3C90': (0.042177, 1.5424, 2.6152),
    '3C94': (0.012263, 1.6159, 2.4982),
    '3E6': (0.00015324, 1.9098, 2.0903),
    '3F4': (0.75798, 1.4146, 3.1455),
    '77': (0.053696, 1.5269, 2.519),
    '78': (0.016878, 1.609, 2.5432),
    'N30': (0.0001319, 1.9629, 2.3541),
}

material_names = list(materials.keys())
excitations = ('Datasheet', 'Sinusoidal', 'Triangular', 'Trapezoidal')
prediction_algorithms = ('iGSE', 'Advanced Analytical', 'Machine Learning')
input_dir = "src/magnet/data/"
