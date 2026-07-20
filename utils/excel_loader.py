import pandas as pd


def _normalize_columns(cols):
    return [str(c).strip() for c in cols]


def _standardize_excel_columns(df):
    df.columns = _normalize_columns(df.columns)
    rename_map = {}

    for c in df.columns:
        cl = str(c).lower().strip().replace("_", " ").replace("-", " ")
        cl = " ".join(cl.split())

        if cl in ["dip direction", "dipdirection", "dip dir", "dipdir",
                  "dip direct", "dip directi"]:
            rename_map[c] = "Dip Direction"
        elif cl == "dip":
            rename_map[c] = "Dip"
        elif cl in ["corrected set", "correctedset"]:
            rename_map[c] = "Corrected Set"
        elif cl == "set":
            rename_map[c] = "Set"
        elif cl in ["family", "fam", "fracture set"]:
            rename_map[c] = "Corrected Set"

    return df.rename(columns=rename_map)


def load_orientations_from_excel(excel_path, sheet=0, valid_families=None):
    """
    Load dip/dip-direction orientations from an Excel file.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file.
    sheet : int or str
        Sheet index or name.
    valid_families : list of int, optional
        If given, only rows whose 'Corrected Set' is in this list are kept.

    Returns
    -------
    orient_by_fam : dict  {fam_id (int) -> DataFrame with ['Dip', 'Dip Direction']}
    """
    df = pd.read_excel(excel_path, sheet_name=sheet)
    df = _standardize_excel_columns(df)

    # Accept plain "Set" when "Corrected Set" is absent OR entirely empty (all NaN)
    if "Corrected Set" in df.columns and df["Corrected Set"].isna().all() and "Set" in df.columns:
        df["Corrected Set"] = df["Set"]
    elif "Corrected Set" not in df.columns and "Set" in df.columns:
        df = df.rename(columns={"Set": "Corrected Set"})

    required = {"Dip", "Dip Direction", "Corrected Set"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Excel must contain columns {required}, found {list(df.columns)}"
        )

    df["Dip"] = pd.to_numeric(df["Dip"], errors="coerce")
    df["Dip Direction"] = pd.to_numeric(df["Dip Direction"], errors="coerce") % 360.0
    df["Corrected Set"] = pd.to_numeric(df["Corrected Set"], errors="coerce")
    df = df.dropna(subset=["Dip", "Dip Direction", "Corrected Set"]).copy()
    df["Corrected Set"] = df["Corrected Set"].astype(int)

    if valid_families is not None:
        df = df[df["Corrected Set"].isin(valid_families)].copy()

    orient_by_fam = {
        int(fam_id): g[["Dip", "Dip Direction"]].reset_index(drop=True)
        for fam_id, g in df.groupby("Corrected Set")
    }
    return orient_by_fam