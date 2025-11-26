"""
ML pipeline for 2D-like materials from OQMD:

- Query OQMD /formationenergy for entries with band_gap, sites, unit_cell
- Clean data and restrict to "2D-like" chemistries (TMDC-ish: TM + chalcogen)
- Featurizer 1 (composition): avg_atomic_number, avg_atomic_mass, ntypes
- Featurizer 2 (structure): a, b, c, volume, density, c/a, b/a
- Train RandomForestRegressor to predict band_gap
- Evaluate and plot parity plots
"""

import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pymatgen.core import Composition, Element, Structure, Lattice

plt.style.use("seaborn-v0_8-darkgrid")

# ============================================
# OQMD REST helpers
# ============================================

BASE_URL = "http://oqmd.org/oqmdapi"
DEFAULT_TIMEOUT = 30  # seconds


def build_url(resource: str) -> str:
    return f"{BASE_URL}/{resource}"


def fetch_oqmd_page(
    resource: str,
    params: Dict[str, Any],
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    url = build_url(resource)
    sess = session or requests.Session()
    resp = sess.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_oqmd_all(
    resource: str,
    base_params: Dict[str, Any],
    max_records: Optional[int] = None,
    page_limit: int = 200,
    sleep: float = 0.2,
) -> List[Dict[str, Any]]:
    """Fetch multiple pages from OQMD."""
    all_results: List[Dict[str, Any]] = []
    offset = 0

    with requests.Session() as sess:
        while True:
            params = base_params.copy()
            params["limit"] = page_limit
            params["offset"] = offset
            params.setdefault("format", "json")

            print(
                f"[INFO] Fetching offset={offset} (page size={page_limit})...")
            data = fetch_oqmd_page(resource, params, session=sess)

            if isinstance(data, dict) and "data" in data:
                page_results = data["data"]
            elif isinstance(data, list):
                page_results = data
            else:
                raise ValueError("Unexpected OQMD response structure")

            if not page_results:
                print("[INFO] No more results returned.")
                break

            all_results.extend(page_results)
            offset += len(page_results)

            if max_records is not None and len(all_results) >= max_records:
                print(f"[INFO] Reached max_records={max_records}.")
                all_results = all_results[:max_records]
                break

            time.sleep(sleep)

    return all_results


# ============================================
# Structure builder for OQMD sites/unit_cell
# ============================================

def oqmd_sites_unitcell_to_structure(
    sites: List[str],
    unit_cell: List[List[float]],
) -> Optional[Structure]:
    """
    Build a pymatgen Structure from OQMD 'sites' and 'unit_cell'.

    Example site string: "Ge @ 0 0.5 0.5"
    """
    try:
        lattice = Lattice(unit_cell)

        species = []
        coords = []

        for s in sites:
            parts = s.split()
            # expect: [element, "@", x, y, z]
            if len(parts) != 5 or parts[1] != "@":
                print(f"[WARN] Unexpected site format: {s}")
                continue
            elem = parts[0]
            x, y, z = map(float, parts[2:])
            species.append(elem)
            coords.append([x, y, z])

        if not species:
            return None

        return Structure(lattice, species, coords, coords_are_cartesian=False)
    except Exception as e:
        print(f"[WARN] Failed to build Structure from sites/unit_cell: {e}")
        return None


# ============================================
# Featurizer 1: composition stats
# ============================================

def composition_stats(formula: str) -> Dict[str, float]:
    """
    Composition-based features:
      - avg_atomic_number
      - avg_atomic_mass
      - ntypes (distinct elements)
    """
    comp = Composition(formula)
    total_atoms = comp.num_atoms

    zs = []
    masses = []
    for el, amt in comp.get_el_amt_dict().items():
        frac = amt / total_atoms
        elem = Element(el)
        zs.append(elem.Z * frac)
        masses.append(elem.atomic_mass * frac)

    avg_z = float(sum(zs))
    avg_mass = float(sum(masses))
    ntypes = float(len(comp.elements))

    return {
        "avg_atomic_number": avg_z,
        "avg_atomic_mass": avg_mass,
        "ntypes": ntypes,
    }


# ============================================
# Featurizer 2: simple structural features
# ============================================

def structural_features(structure: Structure) -> Dict[str, float]:
    """
    Simple structural features from the lattice:
      - a, b, c
      - volume
      - density
      - c/a, b/a
    """
    lat = structure.lattice
    a, b, c = lat.a, lat.b, lat.c
    vol = lat.volume
    density = structure.density  # g/cm^3

    # avoid divide-by-zero just in case
    c_over_a = c / a if a != 0 else 0.0
    b_over_a = b / a if a != 0 else 0.0

    return {
        "a": a,
        "b": b,
        "c": c,
        "volume": vol,
        "density": density,
        "c_over_a": c_over_a,
        "b_over_a": b_over_a,
    }


# ============================================
# Main pipeline
# ============================================

def main():
    # -------- 1. Query OQMD with structural info --------
    resource = "formationenergy"

    base_params = {
        # name=formula, band_gap=target, sites/unit_cell for structure
        "fields": "name,entry_id,band_gap,sites,unit_cell",
        "filter": "",
    }

    max_records = 5000  # adjust as needed

    print("[INFO] Querying OQMD...")
    results = fetch_oqmd_all(
        resource=resource,
        base_params=base_params,
        max_records=max_records,
        page_limit=200,
    )

    df_raw = pd.DataFrame(results)
    print("Raw data shape:", df_raw.shape)
    if df_raw.empty:
        print("[ERROR] No data returned from OQMD.")
        return

    # -------- 2. Basic cleaning --------
    df = df_raw.copy()

    # require band_gap, name, sites, unit_cell
    df = df.dropna(subset=["band_gap", "name", "sites", "unit_cell"])
    df = df[df["band_gap"] >= 0]
    df = df.drop_duplicates(subset=["name"])

    df = df.rename(columns={"name": "formula"})
    print("Cleaned (pre-structure) shape:", df.shape)

    # -------- 3. Optional: 2D-like chemistry filter (TMDC-ish) --------
    tm_elements = {"Ti", "V", "Cr", "Mn", "Fe",
                   "Co", "Ni", "Mo", "W", "Zr", "Hf"}
    chalcogens = {"O", "S", "Se", "Te"}

    def is_2d_like_formula(formula: str) -> bool:
        comp = Composition(formula)
        elems = {el.symbol for el in comp.elements}
        return len(elems & chalcogens) > 0

    df["is_2d_like"] = df["formula"].apply(is_2d_like_formula)
    df = df[df["is_2d_like"]].drop(columns=["is_2d_like"])

    print("After 2D-like chemistry filter:", df.shape)
    if df.empty:
        print("[ERROR] No rows left after 2D-like filter. Loosen criteria.")
        return

    # -------- 4. Build structures --------
    print("[INFO] Building pymatgen.Structure objects...")
    df["structure"] = df.apply(
        lambda row: oqmd_sites_unitcell_to_structure(
            row["sites"], row["unit_cell"]),
        axis=1,
    )
    df = df.dropna(subset=["structure"])
    print("After dropping invalid structures:", df.shape)

    # -------- 5. Composition features --------
    print("[INFO] Computing composition statistics...")
    stats_series = df["formula"].apply(composition_stats)
    stats_df = pd.DataFrame(stats_series.tolist(), index=df.index)

    # -------- 6. Structural features --------
    print("[INFO] Computing structural features...")
    struct_series = df["structure"].apply(structural_features)
    struct_df = pd.DataFrame(struct_series.tolist(), index=df.index)

    # Combine all features into df_feat
    df_feat = pd.concat([df, stats_df, struct_df], axis=1)

    comp_cols = ["avg_atomic_number", "avg_atomic_mass", "ntypes"]
    struct_cols = ["a", "b", "c", "volume", "density", "c_over_a", "b_over_a"]

    # -------- 7. Build X, y --------
    X = df_feat[comp_cols + struct_cols].values
    y = df_feat["band_gap"].values

    print("Feature matrix shape:", X.shape)

    # -------- 8. Scaling + splits --------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_fraction = 0.1
    validation_fraction = 0.2

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=test_fraction, random_state=17
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=validation_fraction / (1 - test_fraction),
        random_state=17,
    )

    print(
        f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # -------- 9. Baseline model --------
    mean_train = y_train.mean()
    baseline_mae = mean_absolute_error(
        y_val, np.full_like(y_val, fill_value=mean_train))
    print(f"Baseline MAE (predicting mean band gap): {baseline_mae:.4f} eV")

    # -------- 10. RandomForest hyperparameter sweep --------
    n_estimators_list = [50, 100, 200, 400]
    train_maes = []
    val_maes = []

    for n in n_estimators_list:
        rf = RandomForestRegressor(
            n_estimators=n,
            random_state=17,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_train_pred = rf.predict(X_train)
        y_val_pred = rf.predict(X_val)

        train_maes.append(mean_absolute_error(y_train, y_train_pred))
        val_maes.append(mean_absolute_error(y_val, y_val_pred))

    plt.figure(figsize=(6, 4))
    plt.plot(n_estimators_list, train_maes, "o-", label="Train MAE")
    plt.plot(n_estimators_list, val_maes, "o-", label="Val MAE")
    plt.xlabel("n_estimators")
    plt.ylabel("MAE (eV)")
    plt.legend()
    plt.title("RF performance vs n_estimators")
    plt.tight_layout()
    plt.show()

    best_idx = int(np.argmin(val_maes))
    best_n_estimators = n_estimators_list[best_idx]
    print(
        f"Best n_estimators = {best_n_estimators} (Val MAE = {val_maes[best_idx]:.4f} eV)")

    # -------- 11. Final RF model + evaluation --------
    rf_final = RandomForestRegressor(
        n_estimators=best_n_estimators,
        random_state=17,
        n_jobs=-1,
    )
    rf_final.fit(X_train, y_train)

    y_train_pred = rf_final.predict(X_train)
    y_val_pred = rf_final.predict(X_val)
    y_test_pred = rf_final.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("=== Final RF Performance (comp + structural features) ===")
    print(f"Train: MAE={mae_train:.3f}, R2={r2_train:.3f}")
    print(f"Val:   MAE={mae_val:.3f}, R2={r2_val:.3f}")
    print(f"Test:  MAE={mae_test:.3f}, R2={r2_test:.3f}")

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

    parity_plot(y_train, y_train_pred,
                f"Train (MAE={mae_train:.3f}, R2={r2_train:.3f})")
    parity_plot(y_val,   y_val_pred,
                f"Val (MAE={mae_val:.3f}, R2={r2_val:.3f})")
    parity_plot(y_test,  y_test_pred,
                f"Test (MAE={mae_test:.3f}, R2={r2_test:.3f})")

    # -------- 12. Save dataset with features --------
    out_cols = ["entry_id", "formula", "band_gap"] + comp_cols + struct_cols
    df_feat[out_cols].to_csv("oqmd_2d_band_gap_features.csv", index=False)
    print("Saved dataset to 'oqmd_2d_band_gap_features.csv'")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
