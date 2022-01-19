import os.path
import glob
import json
import pandas as pd

from magnet.io import h5_store

INPUT_JSON_DIR = '../src/magnet/data'
OUTPUT_H5_DIR = '../src/magnet/data'

if __name__ == '__main__':

    # str in the JSON files should not contain commas
    cols_meas = {
        'Material': str,
        'Core_Shape': str,
        'Effective_Area': float,
        'Effective_Volume': float,
        'Effective_Length': float,
        'Primary_Turns': int,
        'Secondary_Turns': int,
        'Excitation_Type': str,
        'Duty_1': list,
        'Duty_2': list,
        'Duty_3': list,
        'Duty_4': list,
        'Frequency': list,
        'Flux_Density': list,
        'Power_Loss': list,
        'Outlier_Factor': list,
        'Info_Date': str,
        'Info_Excitation': str,
        'Info_Core': str,
        'Info_Setup': str,
        'Info_Scope': str,
        'Info_Volt_Meas': str,
        'Info_Curr_Meas': str,
    }
    df_cols_meas = {
        'Frequency': float,
        'Flux_Density': float,
        'Duty_1': float,
        'Duty_2': float,
        'Duty_3': float,
        'Duty_4': float,
        'Outlier_Factor': float,
        'Power_Loss': float
    }

    cols_datasheet = {
        'Material': str,
        'Excitation': str,
        'Frequency': list,
        'Flux_Density': list,
        'Temperature': list,
        'Power_Loss': list,
    }
    df_cols_datasheet = {
        'Frequency': float,
        'Flux_Density': float,
        'Temperature': float,
        'Power_Loss': float
    }

    cols_interpolated = {
        'Material': str,
        'Core_Shape': str,
        'Effective_Area': float,
        'Effective_Volume': float,
        'Effective_Length': float,
        'Primary_Turns': int,
        'Secondary_Turns': int,
        'Excitation': str,
        'Frequency': list,
        'Flux_Density': list,
        'Power_Loss': list,
    }
    df_cols_interpolated = {
        'Frequency': float,
        'Flux_Density': float,
        'Power_Loss': float
    }

    cols_datasheet_interpolated = {
        'Material': str,
        'Excitation': str,
        'Frequency': list,
        'Flux_Density': list,
        'Power_Loss': list,
    }

    for filename in glob.glob(f'{INPUT_JSON_DIR}/*Webpage.json'):
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
                assert k in cols_meas
                if not isinstance(d[k], cols_meas[k]):
                    # Any values that don't conform to our expected data types are being stored as 0-length lists
                    assert isinstance(d[k], list) and len(d[k]) == 0

            material = d['Material'].lower()
            excitation = (d['Excitation_Type']).lower()
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
                info_date=d['Info_Date'] or None,
                info_excitation=d['Info_Excitation'] or None,
                info_core=d['Info_Core'] or None,
                info_setup=d['Info_Setup'] or None,
                info_scope=d['Info_Scope'] or None,
                info_volt_meas=d['Info_Volt_Meas'] or None,
                info_curr_meas=d['Info_Curr_Meas'] or None,
            )

            df = pd.DataFrame({k: d[k] for k in df_cols_meas}).astype(df_cols_meas)

            output_filename = f'{material.upper()}_{excitation}.h5'
            h5_store(os.path.join(OUTPUT_H5_DIR, output_filename), df, **m)

    for filename in glob.glob(f'{INPUT_JSON_DIR}/*Datasheet.json'):
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
                assert k in cols_datasheet
                if not isinstance(d[k], cols_datasheet[k]):
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

            df = pd.DataFrame({k: d[k] for k in df_cols_datasheet}).astype(df_cols_datasheet)

            output_filename = f'{material.upper()}_{excitation}.h5'
            h5_store(os.path.join(OUTPUT_H5_DIR, output_filename), df, **m)

    for filename in glob.glob(f'{INPUT_JSON_DIR}/*Sinusoidal_Interpolated.json'):
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
                assert k in cols_interpolated
                if not isinstance(d[k], cols_interpolated[k]):
                    # Any values that don't conform to our expected data types are being stored as 0-length lists
                    assert isinstance(d[k], list) and len(d[k]) == 0

            material = d['Material'].lower()
            excitation = (d['Excitation']).lower()
            assert filename.lower().endswith(f'{material}_{excitation}_interpolated.json')
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

            df = pd.DataFrame({k: d[k] for k in df_cols_interpolated}).astype(df_cols_interpolated)

            output_filename = f'{material.upper()}_{excitation}_interpolated.h5'
            h5_store(os.path.join(OUTPUT_H5_DIR, output_filename), df, **m)

    for filename in glob.glob(f'{INPUT_JSON_DIR}/*Datasheet_Interpolated.json'):
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
                assert k in cols_datasheet_interpolated
                if not isinstance(d[k], cols_datasheet_interpolated[k]):
                    # Any values that don't conform to our expected data types are being stored as 0-length lists
                    assert isinstance(d[k], list) and len(d[k]) == 0

            material = d['Material'].lower()
            excitation = (d['Excitation']).lower()
            assert filename.lower().endswith(f'{material}_{excitation}_interpolated.json')
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

            df = pd.DataFrame({k: d[k] for k in df_cols_interpolated}).astype(df_cols_interpolated)

            output_filename = f'{material.upper()}_{excitation}_interpolated.h5'
            h5_store(os.path.join(OUTPUT_H5_DIR, output_filename), df, **m)