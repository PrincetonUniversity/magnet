materials = {
    # material => (k_i, alpha, beta)
    '3C90': (0.23732, 1.3932, 2.5481),
    '3C94': (2.0046, 1.4361, 2.4674),
    '3E6': (0.00029059, 1.8702, 2.1475),
    '3F4': (52.6956, 1.0598, 2.7734),
    '77': (0.21406, 1.4182, 2.4746),
    '78': (0.095863, 1.4742, 2.4951),
    'N27': (0.42941, 1.3697, 2.4634),
    'N30': (0.00034663, 1.8984, 2.4024),
    'N49': (1.9502, 1.2553, 2.8231),
    'N87': (0.79822, 1.3453, 2.5752),
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
    'N87': 'R34.0X20.5X12.5',
}
material_names = list(materials.keys())
