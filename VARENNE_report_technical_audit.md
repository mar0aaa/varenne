# Varenne KCO/KCO* paper — technical audit against the implemented code

**Audit basis:** current `kco_model.py`, `kco_star_model.py`, `run_kco_varenne_demo.py`, and the CSV/figure outputs in `outputs/VARENNE/10_kco_comparison/`. The report images are screenshots; exact prose phrasing cannot be OCR'd, but the equations, parameter values, and methodology were checked line-by-line against the current code.

---

## A. Correct as written (consistent with the current code and cited literature)

1. **Median size equation `X50`** — `kco_model.py:738-792`
   - `X50 = g(n) A Q^{1/6} q^{-0.8} (115/s_ANFO)^{19/30}` is the Ouchterlony (2005) eq. (11b) / Cunningham (1987) form as written in Ouchterlony & Sanchidrián (2019) eq. (24). Correct.
   - The code returns `X50` in cm and multiplies by 10 for mm; the report should be in mm.

2. **Swebrec cumulative passing equation** — `kco_model.py:1023-1062`
   - `P(x) = 100 / (1 + [ ln(Xmax/x) / ln(Xmax/X50) ]^b )` for `0 < x < Xmax`; `P(x<=0)=0`, `P(x>=Xmax)=100` is Ouchterlony (2005) eq. (11a). Correct.

3. **Undulation parameter `b`** — `kco_model.py:985-1017`
   - `b = 2 ln(2) n ln(Xmax/X50)` is the first line of Ouchterlony (2005) eq. (11c). Correct.

4. **Maximum fragment size `Xmax`** — `kco_model.py:942-979`
   - `Xmax = min(in-situ block size, B, S)` is Ouchterlony (2005) eq. (11d). Correct.

5. **Uniformity index `n`** — `kco_model.py:591-675`
   - `n = n_s sqrt(2 - 30B/d) sqrt((1+S/B)/2) (1 - W/B) (L/H)^0.3 (A/6)^0.3` is the Cunningham (2005) / Ouchterlony & Sanchidrián (2019) eq. (48) form. Correct; the `(A/6)^0.3` term is the 2019 correction.

6. **Rock factor `A` and blastability index `BI` conceptual form**
   - `A = 0.06 BI` is correct (Cunningham 1987; `kco_model.py:531-549`).
   - `RDI = 0.025 rho - 50` is correct (`kco_model.py:403-425`).
   - `HF` with `E < 50 GPa -> E/3` and `E >= 50 GPa -> UCS/5` is correct for `E=60`, `UCS=200` giving `HF=40` (`kco_model.py:428-456`).

7. **Joint factor `JF = JPS + JPA`** — `kco_model.py:372-397`
   - This is the Cunningham (1987) baseline without joint condition factor. Correct.

8. **JPS thresholds** — `kco_model.py:308-369` (corrected to the supervisor's slide)
   - `JPS = 10` if `Sj < 0.1 m`; `20` if `0.1 <= Sj < 0.3 m`; `80` if `0.3 <= Sj < 0.95 sqrt(BS)`; `50` if `Sj > sqrt(BS)`.
   - For `B=3.4 m`, `S=4.1 m`, `0.95 sqrt(BS) = 3.547 m` and `sqrt(BS) = 3.734 m`. The band `0.95 sqrt(BS) <= Sj <= sqrt(BS)` is left undefined by the slide and raises a `ValueError` (flagged, no invented rule).

9. **JPA mapping for `cunningham_2005`** — `kco_model.py:141-184`
   - `strike_perpendicular_to_face -> 30`. Correct with the active 2005 mapping.

10. **`S_j^* = V^{1/3}` equivalent-cube conversion** — `kco_model.py:798-815`, `kco_star_model.py:118-172`
    - Correct. The report should, and does in the equations, present it as the equivalent-cube side of a DFN block. It is a KCO* modelling choice; KCO itself uses a single mean `S_j`.

11. **12-class probabilistic construction (log bins p1-p99 tails)** — `kco_star_model.py:489-641`
    - 2 tails + 10 central log-spaced bins; 2303 blocks; weights sum to 1.00. Verified in the generated `VARENNE_kco_star_section3_table.csv`.
    - Representative `S_{j,i}^*` is the median of the original metre values. Correct.
    - `ln(S_j^*)` is used **only** for the 10 central bin edges; KCO is run on metre values. Correct.

12. **Weighted mixture `P_{KCO*}(x) = sum_i w_i P_i(x)`** — `kco_star_model.py:722-791`
    - Mathematically implemented exactly as a probability-weighted sum of the per-class Swebrec passing curves. Conceptually this is a **new KCO* proposal**, not an existing KCO equation; as long as the paper states this clearly, it is acceptable.

13. **Varenne blast geometry inputs**
    - `B = 3.4 m`, `S = 4.1 m`, `H = 14 m`, `D = 114 mm`, `Q = 165 kg/hole` are the values in `run_kco_varenne_demo.py:131-152`. Correct.

14. **Corrected rock properties**
    - `rho = 2700 kg/m3`, `UCS = 200 MPa`, `E = 60 GPa` as used in `tmp_section3_table.py` and the professor-provided values. Correct.

15. **Collapse of 12 classes into 3 distinct `X50` values**
    - The `VARENNE_sj_star_log_bins_table_volume_weighted_xmaxSBH_jps80.csv` shows three distinct `X50` values: 178.94 mm (JPS=10), 197.29 mm (JPS=20), 307.41 mm (JPS=80). This is a direct consequence of JPS being a step function. Correct.

16. **KCO/KCO* are much finer than WipFrag**
    - WipFrag `D50` ~540.75 mm; KCO `X50` = 307.41 mm; KCO* `X50*` = 307.36 mm. Both models are clearly finer than the measurement. Correct interpretation.

17. **`n_s = 1` and `g(n) = 1` are the baseline, no-shift, no-timing-scatter assumptions** — `kco_model.py:686-715`
    - Correct as placeholders. They should be described as defaults/assumptions, not as calibrated values.

---

## B. Must be corrected

1. **Blastability-index equation in the paper is currently written as `BI = RMD + JF + RDI + HF`. This is incorrect and double-counts `JF`.**
   - In the current code, `BI = RMD + RDI + HF` (`kco_model.py:505-528`), and for jointed rock `RMD = JF` (`kco_model.py:462-502`) where `JF = JPS + JPA`.
   - Adding `JF` again as a separate summand is a mixing of the `RMD` and `JF` definitions.
   - **D. Replace by:**
     ```
     BI = RMD + RDI + HF
     where RMD = JF for jointed rock, JF = JPS + JPA
     ```
     This is the form in Ouchterlony (2005) eq. (11e) and the code; `RMD` already contains the jointing information for the jointed case.

2. **The main comparison figure `VARENNE_kco_vs_kco_star_comparison.png` and the D-table are generated by `run_kco_varenne_demo.py` with `ucs_mpa = 100.0`, not 200.0, so they do NOT match the `UCS=200` values reported in the text.**
   - `run_kco_varenne_demo.py:151` still reads `ucs_mpa=100.0`.
   - With `UCS=100`, `HF=20`, `A=7.05`, and the code produces `X50` ≈ 215.6 mm, not the `252.35` mm listed in the report.
   - When I regenerated the PNG for the legend move, it was with `UCS=100` and `X50` ≈ 215 mm.
   - **D. Replace / action:**
     - In `run_kco_varenne_demo.py` line 151, change `ucs_mpa=100.0` to `ucs_mpa=200.0`.
     - Re-run `run_kco_varenne_demo.py` to regenerate `VARENNE_kco_vs_kco_star_comparison.png` and `VARENNE_D20_D50_D80_D90_table.csv`.
     - Or, if the PNG in the paper must use the already-corrected script, regenerate it from a script that explicitly uses `UCS=200`.

3. **`s_ANFO = 100 %` must not be described as a confirmed XL900 property.**
   - `run_kco_varenne_demo.py:148` sets `s_anfo_pct=100.0` and comments it as an "ESTIMATE".
   - No product data sheet or RWS for Dyno Nobel XL900 was found in any project file.
   - **D. Replace any statement implying confirmation by:**
     ```
     The explosive is referred to as Dyno Nobel XL900 emulsion, but its
     relative weight strength was not available in the project files.
     Therefore s_ANFO = 100 % is used as an ANFO-equivalent placeholder.
     ```

4. **Reported vs calculated powder factor `q`.**
   - The geometric calculation from `Q/(B S H)` is `165 / (3.4 * 4.1 * 14) = 0.845 kg/m3`.
   - The KCO/KCO* runs use the **reported** `q = 0.80 kg/m3` (`powder_factor_mode="reported"` in `predict_kco_star`).
   - **D. The text must explicitly state that 0.80 kg/m3 is the value used in the KCO calculations, while 0.845 kg/m3 is an independent check.**

5. **The weighted-mixture `P_{KCO*}` must be explicitly described as the KCO* methodology being proposed, not a published KCO equation.**
   - `kco_star_model.py:722-734` docstring flags it as "a KCO* methodology proposed for testing; it is NOT a published Cunningham/Ouchterlony equation."
   - **D. Add a sentence:** "The weighted mixture is a methodological extension introduced in this study; it is not part of the original KCO/Kuz–Ram equations."

---

## C. Needs verification / cannot be fully confirmed from the available material

1. **Exact KCO* `X50*` for `UCS=200` (current run: volume-weighted, Xmax from the blast-cell DFN)**
   - Current values (`outputs/VARENNE/10_kco_comparison/VARENNE_kco_results_summary_volume_weighted_xmaxSBH_jps80.md`): class `X50` = 178.94, 197.29, 307.41 mm (JPS 10/20/80) with VOLUME weights 0.0002 %, 0.039 %, 99.960 %; mixture `X50* = 307.36 mm`, D20/D80/D90 = 105.76 / 612.60 / 813.84 mm.
   - Superseded report values: 245.56 mm (count-weighted, Xmax 1813.48 mm) and 208.77 mm (UCS=100). The paper must state `xmax_block_percentile` (95), `class_method` (log_bins_p1_p99_tails) and `weight_method` (volume).

2. **WipFrag `D50 = 540.75 mm` and in-situ DFN `D50 ≈ 1024 mm`**
   - These are from project data files; they should be checked against the exact WipFrag curve and the DFN block-volume distribution outputs. They were not re-run in this audit.

3. **`Xmax` for the comparison curves**
   - CURRENT: `Xmax = min(P95(Sj*), B, S) = min(2.16225, 3.4, 4.1) m = 2162.25 mm` for every class and for classical KCO, where `Sj* = V^(1/3)` of the SAME pooled 30-realization S x B x H blast-cell DFN (2303 blocks) used for the Sj* classes (`run_kco_varenne_demo.py`, `xmax_vols = sj_star_vols`).
   - SUPERSEDED: `Xmax = 1813.48 mm` (P95 of the 50x50x50 m DFN, `block_shape_data_VARENNE.csv`). Any paper value derived from it (Swebrec `b` = 3.028/3.164/3.208, D20/D80/D90 = 80.3/520.7/698.1 mm) is obsolete; current `b` = 3.178 / 3.415 / 3.451 (JPS 80/20/10).
   - The 50x50x50 m DFN remains in the methodology ONLY for the P32 calibration of the fracture families.

4. **JPA mapping version**
   - The active code uses `jpa_mapping_version="cunningham_2005"` (`_joint_plane_angle_rating` default). If the paper cites Cunningham (1987) for `JPA=30`, the mapping is reversed for `dip_out_of_face` and `dip_into_face`. The paper should state that the 2005 mapping is used.

5. **`Xmax_block_percentile` and `block_size_method` for `Xmax`**
   - These are user choices. The paper should report them so the figure/table is reproducible. Recommended: "`Xmax = min(P95(Sj*), B, S)`, where `Sj* = V^(1/3)` is computed for the pooled blocks of the 30 blast-cell (S x B x H) DFN realizations, i.e. the same population that defines the Sj* classes; Xmax = 2162.25 mm, governed by the in-situ block size."

6. **Number of DFN realizations and pooled blocks**
   - The `VARENNE_kco_star_section3_table` confirms `n_retained = 2303` blocks. The 30-realization claim should be verified from `generate_blast_cell_realizations` output; it is not contradicted by the available data.

---

## D. Exact replacement text / equations where needed

### D1. Blastability index

**Current (incorrect):**
```
BI = RMD + JF + RDI + HF
```

**Replace by:**
```
For the jointed rock-mass case used at Varenne:
  JF = JPS + JPA
  RMD = JF
  BI = RMD + RDI + HF

with
  RDI = 0.025 rho - 50
  HF = UCS / 5   because E = 60 GPa >= 50 GPa
```

*(Source: `kco_model.py:505-528`, `rock_mass_description`, `blastability_index`; Ouchterlony 2005, eq. (11e).)*

### D2. Explosive strength

**Current (if it implies confirmation):** Any sentence reading as if 100 % is the known XL900 RWS.

**Replace by:**
```
The ANFO-equivalent weight strength is not known for the XL900
emulsion used at Varenne. In the absence of a product-specific RWS,
s_ANFO = 100 % (ANFO-equivalent) is retained as a placeholder.
```

### D3. Powder factor

**Replace / clarify by:**
```
The mine reported a powder factor of q = 0.80 kg/m3 of rock.
Independently, from the charge per hole and the blast geometry,
q = Q / (B S H) = 165 / (3.4 x 4.1 x 14) = 0.845 kg/m3.
The reported value 0.80 kg/m3 is retained as the KCO/KCO* input.
```

### D4. `g(n)` and shift factor

**Add:**
```
The shift factor g(n) in the X50 equation is set to 1.0
(no_shift mode), as accepted by Ouchterlony (2005) for the Swebrec
function. This is a baseline assumption, not a calibrated value.
```

### D5. `UCS` in `run_kco_varenne_demo.py`

**Action, not just text:**
```python
# run_kco_varenne_demo.py, line 151
    ucs_mpa=200.0,    # was 100.0; corrected per professor's values
```
Then re-run to regenerate the comparison figure and D-table.

### D6. Weighted mixture

**Add:**
```
The KCO* curve is built as a probability-weighted sum of the
per-class Swebrec curves: P_{KCO*}(x) = sum_i w_i P_i(x). This is a
methodological extension introduced in the present study; it is not
part of the original Cunningham (1987, 2005) or Ouchterlony (2005)
KCO formulation.
```

---

## E. Literature-to-code mapping

| Source as cited | What the code uses it for | File:line |
|---|---|---|
| Cunningham (1987) | `A = 0.06 BI`; `JF = JPS + JPA`; JPS thresholds | `kco_model.py:531-549`, `372-397`, `308-369` |
| Cunningham (2005) | Uniformity index `n` (with 2019 correction) | `kco_model.py:591-675` |
| Ouchterlony (2005) | `X50` eq. (11b), Swebrec eq. (11a), `b` eq. (11c), `Xmax` eq. (11d), `RMD` eq. (11e) | `kco_model.py:738-792`, `1023-1062`, `985-1017`, `942-979`, `462-502` |
| Ouchterlony & Sanchidrián (2019) | Corrected `n` with `(A/6)^0.3` | `kco_model.py:591-675` |
| Lilly (1986) / Cunningham (1987) | `HF` with `E < 50` vs `E >= 50` | `kco_model.py:428-456` |
| KCO* (this study) | `S_j^* = V^{1/3}`, log-bins, probability-weighted mixture | `kco_star_model.py:118-172`, `489-641`, `722-791` |

## F. Highest-priority action before submission

1. Fix the `BI = RMD + JF + RDI + HF` equation to `BI = RMD + RDI + HF` (with `RMD = JF` for jointed rock).
2. Change `ucs_mpa=100.0` to `200.0` in `run_kco_varenne_demo.py` and regenerate the comparison PNG and D-table, otherwise the figure values (≈215 mm) will not match the paper's text (≈252 mm).
3. Disclaim the `s_ANFO = 100 %` value as a placeholder, not a confirmed XL900 property.
4. Clarify that `q = 0.80 kg/m3` is the retained reported value and `0.845 kg/m3` is a geometric check.
