import os.path
import glob
import json
import pandas as pd

from magnet.io import h5_store

INPUT_JSON_DIR = '../src/magnet/data/datasheet'
OUTPUT_H5_DIR = '../src/magnet/data'


if __name__ == '__main__':

    cols = {
        'Material': str,
        'Excitation': str,
        'Frequency': list,
        'Flux_Density': list,
        'Temperature': list,
        'Power_Loss': list,
    }

    df_cols = {
        'Frequency': float,
        'Flux_Density': float,
        'Temperature': float,
        'Power_Loss': float
    }

    for filename in glob.glob(f'{INPUT_JSON_DIR}/*.json'):
        with open(filename) as f:
            d = json.load(f)

            # -------------------------
            # DATA VALIDATION
            # -------------------------
            # For any values that are lists of non-zero length, they should be the same size
            unique_lengths = set([len(d[k]) for k in d.keys() if isinstance(d[k], list)])
            if 0 in unique_lengths:
                unique_lengths.remove(0)
            assert len(unique_lengths) == 1

            for k in d.keys():
                assert k in cols
                if not isinstance(d[k], cols[k]):
                    # Any values that don't conform to our expected data types are being stored as 0-length lists
                    assert isinstance(d[k], list) and len(d[k]) == 0

            material = d['Material'].lower()
            excitation = (d['Excitation']).lower()
            assert filename.lower().endswith(f'{material}_{excitation}.json')
            # -------------------------
            # DATA VALIDATION
            # -------------------------

            # ----------
            # Metadata
            # ----------
            m = dict(
                material=material,
                excitation_type=excitation,
            )

            df = pd.DataFrame({k: d[k] for k in df_cols}).astype(df_cols)

            output_filename = f'{material.upper()}_{excitation}.h5'
            h5_store(os.path.join(OUTPUT_H5_DIR, output_filename), df, **m)
