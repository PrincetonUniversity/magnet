import os.path
import glob
import json
import pandas as pd

from magnet.io import h5_store

INPUT_JSON_DIR = '../src/magnet/data'
OUTPUT_H5_DIR = '../src/magnet/data'

if __name__ == '__main__':

    # str in the JSON files should not contain commas
    cols_database = {
        'Material': str,
        'Info_Setup': str,
        'Info_Core': str,
        'Info_Processing': str,
        'Frequency': list,
        'Flux_Density': list,
        'DC_Bias': list,
        'Duty_P': list,
        'Duty_N': list,
        'Temperature': list,
        'Power_Loss': list,
    }

    df_cols_database = {
        'Frequency': float,
        'Flux_Density': float,
        'DC_Bias': float,
        'Duty_P': float,
        'Duty_N': float,
        'Temperature': float,
        'Power_Loss': float
    }

    for filename in glob.glob(f'{INPUT_JSON_DIR}/*database.json'):
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
                assert k in cols_database
                if not isinstance(d[k], cols_database[k]):
                    # Any values that don't conform to our expected data types are being stored as 0-length lists
                    assert isinstance(d[k], list) and len(d[k]) == 0

            material = d['Material'].lower()
            assert filename.lower().endswith(f'{material}_database.json')
            # -------------------------
            # DATA VALIDATION
            # -------------------------

            # ----------
            # Metadata
            # ----------
            m = dict(
                info_setup=d['Info_Setup'] or None,
                info_core=d['Info_Core'] or None,
                info_processing=d['Info_Processing'] or None,
            )

            df = pd.DataFrame({k: d[k] for k in df_cols_database}).astype(df_cols_database)

            output_filename = f'{material.upper()}_database.h5'
            h5_store(os.path.join(OUTPUT_H5_DIR, output_filename), df, **m)
