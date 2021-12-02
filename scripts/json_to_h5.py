import os.path
import glob
import json
import pandas as pd

from magnet.io import h5_store

INPUT_JSON_DIR = '../src/magnet/data'
OUTPUT_H5_DIR = '../src/magnet/data'


if __name__ == '__main__':

    cols = {
        'Material': str,
        'Core_Shape': str,
        'Effective_Area': float,
        'Effective_Volume': float,
        'Effective_Length': float,
        'Primary_Turns': int,
        'Secondary_Turns': int,
        'Excitation_Type': str,
        'Frequency': list,
        'Power_Loss': list,
        'Duty_1': list,
        'Duty_2': list,
        'Duty_3': list,
        'Duty_4': list,
        'Flux_Density': list,
        'Outlier_Factor': list,

    }

    df_cols = {
        'Frequency': float,
        'Flux_Density': float,
        'Duty_1': float,
        'Duty_2': float,
        'Duty_3': float,
        'Duty_4': float,
        'Outlier_Factor': float,
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
            excitation = (d['Excitation_Type'] or 'Datasheet').lower()
            assert filename.lower().endswith(f'{material}_{excitation}_webpage.json')
            # -------------------------
            # DATA VALIDATION
            # -------------------------

            # ----------
            # Metadata
            # ----------
            m = dict(
                material=material,
                core_shape=d['Core_Shape'] or None,
                effective_area=d['Effective_Area'] or None,
                effective_length=d['Effective_Length'] or None,
                effective_volume=d['Effective_Volume'] or None,
                excitation_type=excitation,
                primary_turns=d['Primary_Turns'] or None,
                secondary_turns=d['Secondary_Turns'] or None,
            )

            df = pd.DataFrame({k: d[k] for k in df_cols}).astype(df_cols)

            output_filename = f'{material.upper()}_{excitation}.h5'
            h5_store(os.path.join(OUTPUT_H5_DIR, output_filename), df, **m)