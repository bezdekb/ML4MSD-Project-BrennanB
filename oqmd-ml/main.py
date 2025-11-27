# ML4MSD Final Project
# ME5374
# Brennan Bezdek
# Presenting 12/2/2025

import time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from pymatgen.core import Composition, Element, Structure, Lattice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Tips on how to pull from OQMD API online (https://static.oqmd.org/static/docs/restful.html)
# and general format on what oqmd uses to pull data from (https://github.com/simonverret/materials_data_api_scripts.git)

URL = "http://oqmd.org/oqmdapi"


def build_url(resource):
    return f"{URL}/{resource}"


def fetch_page(resource, params):
    url = build_url(resource)
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


max_records = 8000

# Tutorial pulls from formationenergy then queries


def download_oqmd_data(max_records):
    resource = "formationenergy"
    base_params = {
        "fields": "name,entry_id,band_gap,sites,unit_cell",
        "filter": "",
        "format": "json",
    }

    print(f"Downloading {max_records}")
    all_rows = []
    offset = 0
    page_limit = 200

    while True:
        params = base_params.copy()
        params["limit"] = page_limit
        params["offset"] = offset

        print(f"Fetching offset={offset}")
        data = fetch_page(resource, params)

        if isinstance(data, dict) and "data" in data:
            rows = data["data"]
        else:
            rows = data

        if not rows:
            break

        all_rows.extend(rows)
        offset += len(rows)

        if len(all_rows) >= max_records:
            break

        time.sleep(0.2)  # 5 queries per second

    df_raw = pd.DataFrame(all_rows).head(max_records)
    print(f"Step 1: Downloaded {len(df_raw)} rows from OQMD")
    return df_raw


def string_to_structure(sites, unit_cell):
    try:
        lattice = Lattice(unit_cell)
        species = []
        coords = []

        for s in sites:
            parts = s.split()
            if len(parts) != 5 or parts[1] != "@":
                continue
            elem = parts[0]
            x, y, z = map(float, parts[2:])
            species.append(elem)
            coords.append([x, y, z])

        if not species:
            return None

        structure = Structure(lattice, species, coords,
                              coords_are_cartesian=False)
        return structure
    except Exception:
        return None

# defining 2d materials by transition metal commonly paired with a chalcogen


def material2d(formula):
    tms = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    }  # Transition metals
    ch = {"S", "Se", "Te"}  # Chalcogens

# Checking if these are present
    try:
        composition = Composition(formula)
        elements = {el.symbol for el in composition.elements}
        tm_yes = len(elements & tms) > 0
        ch_yes = len(elements & ch) > 0
        return tm_yes and ch_yes
    except Exception:
        return False


def clean_material2d(df_raw):
    print("Step 2: Data Cleaning")
    df = df_raw.copy()
    df = df.dropna(subset=["band_gap", "name", "sites", "unit_cell"])
    df = df[df["band_gap"] >= 0]
    df = df.drop_duplicates(subset=["name"])
    df = df.rename(columns={"name": "formula"})
    df["structure"] = df.apply(
        lambda row: string_to_structure(row["sites"], row["unit_cell"]),
        axis=1,
    )
    df = df.dropna(subset=["structure"])
    df["chemistry"] = df["formula"].apply(material2d)
    df = df[df["chemistry"]]
    df = df.drop(columns=["chemistry"])
    print(f"Final Dataframe: {len(df)} rows")
    return df

# Composition Featurizer: 3 Features


def composition_stats(formula):
    composition = Composition(formula)
    total_atoms = composition.num_atoms

    average_Z = 0.0
    average_mass = 0.0
    for el, amt in composition.get_el_amt_dict().items():
        fraction = amt / total_atoms
        element = Element(el)
        average_Z += element.Z * fraction
        average_mass += element.atomic_mass * fraction

    stats = {
        "avg_atomic_number": float(average_Z),
        "avg_atomic_mass": float(average_mass),
        "ntypes": float(len(composition.elements)),
    }
    return stats

# Structural Featurizer: 7 Features (lattice, a, b, c, volume, density, c/a, and b/a)


def structural_features(structure):

    lattice = structure.lattice
    a = lattice.a
    b = lattice.b
    c = lattice.c
    volume = lattice.volume
    density = structure.density

    if a != 0:
        c_over_a = c / a
        b_over_a = b / a
    else:
        c_over_a = 0.0
        b_over_a = 0.0

    feats = {
        "a": a,
        "b": b,
        "c": c,
        "volume": volume,
        "density": density,
        "c_over_a": c_over_a,
        "b_over_a": b_over_a,
    }
    return feats


def add_features(df_clean):
    print("Step 3: Featurize")

    composition_series = df_clean["formula"].apply(composition_stats)
    composition_df = pd.DataFrame(
        composition_series.tolist(), index=df_clean.index)

    struct_series = df_clean["structure"].apply(structural_features)
    struct_df = pd.DataFrame(struct_series.tolist(), index=df_clean.index)

    df_featurized = pd.concat([df_clean, composition_df, struct_df], axis=1)
    print(f"Dataframe: {df_featurized.shape}")
    return df_featurized


def train_and_evaluate(df_featurized):
    print("Step 4: Building different data sets")

    feature_cols = [
        "avg_atomic_number", "avg_atomic_mass", "ntypes",
        "a", "b", "c", "volume", "density", "c_over_a", "b_over_a",
    ]
    X = df_featurized[feature_cols].values
    y = df_featurized["band_gap"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 70,20,10 data split
    test_fraction = 0.1
    val_fraction = 0.2

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=test_fraction, random_state=17
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction / (1 - test_fraction),
        random_state=22,
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Val size:   {X_val.shape[0]}")
    print(f"Test size:  {X_test.shape[0]}")

    mean_train = y_train.mean()
    baseline_mae = mean_absolute_error(
        y_val, np.full_like(y_val, fill_value=mean_train))
    print(f"Step 5: Predict Baseline MAE = {baseline_mae:.3f} eV")

    # RandomForest model
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=17,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("Performance Results")
    print(f"Train: MAE={mae_train:.4f}, R2={r2_train:.4f}")
    print(f"Val:   MAE={mae_val:.4f}, R2={r2_val:.4f}")
    print(f"Test:  MAE={mae_test:.4f}, R2={r2_test:.4f}")

    # Parity plots
    def parity_plot(y_true, y_pred, title):
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.figure(figsize=(4.5, 4))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min_val, max_val], [min_val, max_val], "k--")
        plt.xlabel("True band gap (eV)")
        plt.ylabel("Predicted band gap (eV)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    print("Step 6: Plotting")
    parity_plot(y_train, y_train_pred,
                f"Train (MAE={mae_train:.4f}, R2={r2_train:.4f})")
    parity_plot(y_val,   y_val_pred,
                f"Val (MAE={mae_val:.4f}, R2={r2_val:.4f})")
    parity_plot(y_test,  y_test_pred,
                f"Test (MAE={mae_test:.4f}, R2={r2_test:.4f})")


def save_features(df_featurized, filename="2d_materials.csv"):
    cols = [
        "entry_id", "formula", "band_gap",
        "avg_atomic_number", "avg_atomic_mass", "ntypes",
        "a", "b", "c", "volume", "density", "c_over_a", "b_over_a",
    ]
    df_featurized[cols].to_csv(filename, index=False)
    print(f"Step 7: Save data to csv file")


def main():
    df_raw = download_oqmd_data(max_records=8000)
    df_clean = clean_material2d(df_raw)

    if df_clean.empty:
        print("No 2D Materials found")
        return

    df_featurized = add_features(df_clean)
    train_and_evaluate(df_featurized)
    save_features(df_featurized)


if __name__ == "__main__":
    main()
