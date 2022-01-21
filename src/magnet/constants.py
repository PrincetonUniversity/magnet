materials = {
    # material => (k_i, alpha, beta)
    '3C90': (0.042177, 1.5424, 2.6152),
    '3C94': (0.012263, 1.6159, 2.4982),
    '3E6': (0.00015324, 1.9098, 2.0903),
    '3F4': (0.75798, 1.4146, 3.1455),
    '77': (0.053696, 1.5269, 2.519),
    '78': (0.016878, 1.609, 2.5432),
    'N27': (0.066924, 1.5158, 2.5254),
    'N30': (0.0001319, 1.9629, 2.3541),
    'N49': (0.13263, 1.4987, 3.2337),
    'N87': (0.15178, 1.4722, 2.6147),
}

materials_extra = {
    # material => (mu_r_0, f_min, f_max)
    '3C90': (2_300, 25_000, 200_000),
    '3C94': (2_300, 25_000, 300_000),
    '3E6': (10_000, float('nan'), float('nan')),
    '3F4': (900, 25_000, 2_000_000),
    '77': (2_000, 10_000, 100_000),
    '78': (2_300, 25_000, 500_000),
    'N27': (2_000, 25_000, 150_000),
    'N30': (4_300, 10_000, 400_000),
    'N49': (1_500, 300_000, 1_000_000),
    'N87': (2_200, 25_000, 500_000),
}

material_manufacturers = {
    '3C90': 'Ferroxcube',
    '3C94': 'Ferroxcube',
    '3E6': 'Ferroxcube',
    '3F4': 'Ferroxcube',
    '77': 'Fair-Rite',
    '78': 'Fair-Rite',
    'N27': 'TDK',
    'N30': 'TDK',
    'N49': 'TDK',
    'N87': 'TDK',
}

material_applications = {
    '3C90': 'Power and general purpose transformers',
    '3C94': 'Power and general purpose transformers',
    '3E6': 'Wideband transformers and EMI-suppression filters',
    '3F4': 'Power and general purpose transformers',
    '77': 'High and low flux density inductive designs',
    '78': 'Power applications and low loss inductive applications',
    'N27': 'Power transformers',
    'N30': 'Broadband transformers',
    'N49': 'Power transformers',
    'N87': 'Power transformers',
}
material_core_tested = {
    '3C90': 'TX-25-15-10',
    '3C94': 'TX-20-10-7',
    '3E6': 'TX-22-14-6.4',
    '3F4': 'E-32-6-20-R',
    '77': '5977001401',
    '78': '5978007601',
    'N27': 'R20.0X10.0X7.0',
    'N30': 'R22.1X13.7X6.35',
    'N49': 'R16.0X9.6X6.3',
    'N87': 'R22.1X13.7X7.9',
}
material_names = list(materials.keys())
excitations_db = ('Datasheet', 'Sinusoidal', 'Triangular', 'Trapezoidal')
excitations_raw = ('Sinusoidal', 'Triangular-Trapezoidal')
excitations_predict = ('Sinusoidal', 'Triangular', 'Trapezoidal', 'Arbitrary')
# prediction_algorithms = ('iGSE', 'Advanced Analytical', 'Machine Learning')  # Not used, isn't it?
