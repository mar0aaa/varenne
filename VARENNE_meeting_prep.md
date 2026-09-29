# VARENNE : full workflow reconstruction

Everything below was verified against the current code and output files
(run suffix `_volume_weighted_xmaxSBH_jps80_wipfines`). Where something
cannot be proven from the code it says **à confirmer**.

**Main script**: `run_kco_varenne_demo.py`. **Modules**: `kco_model.py`
(published KCO equations), `kco_star_model.py` (KCO* machinery),
`blast_cell_dfn.py` (S×B×H DFN realizations), `run_site.py`
(`SITE_CONFIGS["VARENNE"]`), `main.py` (loaders),
`external_libraries/unblocks` (DFN + block volumes).

**Pipeline**:
```
DFN params (calibrated P32, DIPS orientations, Fisher K, Mauldon sizes)
 -> 30 SxBxH realizations (seeds 3001-3030) -> pooled V (2303 blocks)
 -> Sj* = V^(1/3) -> 12 classes (10 log bins in [P1,P99] + 2 tails)
 -> median Sj* per class -> JPS -> JF -> RMD -> BI -> A -> n -> X50 -> b
 -> 12 Swebrec curves P_i(x)
 -> KCO* = sum w_i^V P_i(x)  (volume weights)
 -> comparison: classical KCO + in-situ CDF + WipFrag
 -> crushed-zone fines corrections (Aubertin + Esen CZI)
```

---

## A. DFN realizations

**1. Purpose** — Build the probabilistic in-situ block-volume population
of ONE blast cell (S×B×H). 30 independent realizations pooled (explicit
assumption: 30 draws of the same cell's statistics, not 30 physical cells).

**2. Inputs**
- `seeds = list(range(3001, 3031))` — `run_kco_varenne_demo.py:215`.
- Geometry `B=3.4, S=4.1, H=14.0` m (`:140-142`), REAL production cell.
  Axis map (`blast_cell_dfn.py:324`): x=S, y=B, z=H.
- Calibrated P32: `outputs/VARENNE/02_calibration/P32_calibrated_summary.csv`,
  column `P32_calibrated`, all flags `ok`: fam1 0.08916, fam2 0.13705,
  fam3 0.11845, fam4 0.07146 (1/m).
- Orientations: `assets/DIPSVARENNE.xlsx` (dip/dipdir poles per family).
- Fracture sizes: Mauldon-corrected lognormal mean/sd via
  `utils.persistence_params.load_size_params` (`assets/varenne_traces.csv`,
  `a_terrain=1480.61 m²`), merged into `SITE_CONFIGS["VARENNE"]["families"]`
  (`run_site.py:42-49`): fam1 fisher 23.88/mean 1.4332/sd 0.5044;
  fam2 30.05/1.6382/0.4045; fam3 32.45/1.3765/0.3048;
  fam4 38.51/1.8158/0.6419 (diameters, m).
- `max_fracs = 6000` per family per realization.

**3. Code**
- `run_kco_varenne_demo.py:203-221`: `REUSE_POOLED_CSV = True` reuses
  `outputs/VARENNE/08_blastcell_SxBxH/block_volumes_blastcell_pooled.csv`
  (col `volume`); else calls `generate_blast_cell_realizations(...)`.
- `blast_cell_dfn.py:239` `generate_blast_cell_realizations` — seed loop.
- `blast_cell_dfn.py:156` `_generate_one_blastcell_dfn` — ONE realization:
  `DFN()` object, region+seed set, 4 fracture sets; per family adds
  circular fractures until `get_P32(i) >= P32_calibrated_i`. Per fracture:
  orientation = sampled field pole + Fisher-K jitter
  (`jitter_orientation`), diameter ~ lognormal, centre uniform in box.
- Blocks (`:340-343`): `Generator().generate_RockMass(dfn)` →
  `gen.get_Volumes(True)` = clean block volumes (m³).

**4. Calculation** — per family: place fractures until volumetric
intensity `P32_i >= P32_calibrated_i`. Lognormal params:
`mu_log = ln(mu²/sqrt(sd²+mu²))`, `sig_log = sqrt(ln(1+(sd/mu)²))`.

**5. Output** — per-seed `BlockVolumes_blastcell_seed<seed>.txt`;
pooled `block_volumes_blastcell_pooled.csv` (cols `seed`,`volume`, 2303
rows); audit `blastcell_config.csv`; in-memory `pooled_volumes_m3`,
`per_seed_volumes_m3`.

Blocks per seed (verified from CSV): 3001→82, 3002→70, 3003→83, 3004→77,
3005→67, 3006→75, 3007→78, 3008→60, 3009→72, 3010→94, 3011→70, 3012→68,
3013→87, 3014→108, 3015→77, 3016→54, 3017→79, 3018→71, 3019→64, 3020→73,
3021→71, 3022→76, 3023→78, 3024→100, 3025→60, 3026→67, 3027→71, 3028→75,
3029→107, 3030→89. **Σ = 2303.**

Pooling code (`blast_cell_dfn.py:333-351`):
```python
for seed in seeds:
    dfn, warns = _generate_one_blastcell_dfn(region_x, region_y, region_z,
        seed, family_ids, families, orient_by_fam, p32_targets, max_fracs)
    gen = Generator(); gen.generate_RockMass(dfn)
    vols = np.array([float(v) for v in gen.get_Volumes(True)])
    per_seed_volumes[int(seed)] = vols
    np.savetxt(seed_txt, vols)
    for v in vols:
        pooled_rows.append({"seed": int(seed), "volume": float(v)})
```

**6. Interpretation** — each realization is one possible fracture network
in the exact blast-cell box, conditioned on calibrated P32, field
orientations, trace statistics. The union approximates the block-size
distribution one cell can contain.


---

## B. Block volume → Sj*

**1. Purpose** — one characteristic length per block for the KCO* classes.

**2. Inputs** — pooled 2303 volumes (m³) = `pooled_df["volume"]`.

**3. Code** — `kco_star_model.py:118` `sj_star_distribution_from_block_volumes`;
conversion `kco_model.py:798` `block_volume_to_equivalent_cube`:
`return np.asarray(volume_m3) ** (1.0/3.0)`.

**4. Calculation** — `Sj*_j = V_j^(1/3)` per block (m). Filter first:
finite & V>0 (`:143-146`). Result: n_retained = 2303, n_rejected = 0 —
all pooled blocks used (Σ N_i = 2303 in the class table).

**5. Output** — `SjStarDistribution`: `block_volume_m3`, `sj_star_m`
(2303 each), counts, summary stats.

**Provenance split**: literature — KCO needs a structural length,
equivalent-cube is an offered conversion; modelling assumption — Sj* ≡
equivalent-cube edge of each DFN block (KCO* methodology, not published);
coded — exactly `V**(1/3)`, no sphere/shape factor.


---

## C. Sj* distribution figure construction

**1. Purpose** — visualise the pooled Sj* population + the class
partition used by KCO*.

**2. Inputs** — `kco_star_result.sj_star_dist.sj_star_m` (2303, m);
60 geomspace display bins; P1/P99; 10 central log bins; KDE on ln(Sj*).

**3. Code** — `run_kco_varenne_demo.py`:
- `_log_bin_percent_hist` (`:849`): `np.histogram` → `pct = 100·N_i/N_tot`
  (assert: no block dropped, sums to 100).
- `:917-929`: `bin_edges = geomspace(min,max,60)` (display only),
  `kde = gaussian_kde(log(sj_sorted_m))`, `pct_kde = kde·Δlnx·100`.
- Fig 3 `:931-974`: bars = % of BLOCKS (count), KDE overlay, JPS-coloured
  `axvspan` per central class, dashed rep-Sj* lines, thick P1/P99 lines →
  `VARENNE_sj_star_histogram_JPS{suffix}.png`.
- Fig 3b `:976-1029`: bars = % of TOTAL VOLUME
  (`np.histogram(..., weights=vol_all)`) — comparison only →
  `..._volume_share{suffix}.png`.

**4. Calculation** — `pct_bin = 100·N_bin/2303`. P1 = `np.percentile(sj,1)`
= 0.037152 m; P99 = `np.percentile(sj,99)` = 2.563949 m.

**5. Output** — 2 PNGs + `VARENNE_sj_star_log_bins_table{suffix}.csv`.

**6. Interpretation** — histogram is descriptive; the same boundaries
become the 12 KCO* classes. The volume-share version shows large blocks
carry most mass (upper classes ≈ 99.96 % of volume).

**7. Say**: « L'histogramme montre la population Sj* poolée en fréquence
de comptage, bins logarithmiques, avec les 10 classes entre P1–P99 et les
2 queues. J'ai aussi la version en part de volume — le volume est dominé
par les gros blocs. »

---

## D. Representative Sj* + class construction

**1. Purpose** — each class needs ONE Sj*_i for the KCO chain.

**2. Inputs** — `n_log_bins=10`, `tail_low_pct=1.0`, `tail_high_pct=99.0`,
`representative_method="median"` (`run_kco_varenne_demo.py:239-259`).

**3. Code** — `kco_star_model.py:489` `create_sj_log_bins_p1_p99_tails`:
- `p_low = percentile(sj_sorted, 1)`, `p_high = percentile(..., 99)`.
- Tails: Sj* < P1 (`is_tail="lower"`), Sj* > P99 (`"upper"`).
- Central: `ln_edges = linspace(log(p_low), log(p_high), 11)` →
  `edges_m = exp(...)` = 10 equal-width ln(Sj*) bins; `np.digitize`.
- `_representative = np.median(values)` (`:552`) — **MEDIAN** of actual
  members (tails included). Asserts: every block once, weights sum to 1.

**4. Calculation** — `Sj*_rep,i = median{Sj*_j ∈ class i}`. Bin-6 example
(class C7): 165 blocks in [0.3086, 0.4713] m → Sj*_rep = 0.3892 m.

**5. Output** — 12 classes:

| Class | Type | Sj* range (m) | N_i | Sj*_rep (m) |
|---|---|---|---|---|
| C1 | lower tail | 0.000257–0.037152 | 24 | 0.0249 |
| C2 | bin 1 | 0.037152–0.056739 | 13 | 0.0511 |
| C3 | bin 2 | 0.056739–0.086650 | 24 | 0.0770 |
| C4 | bin 3 | 0.086650–0.132331 | 25 | 0.1073 |
| C5 | bin 4 | 0.132331–0.202095 | 45 | 0.1698 |
| C6 | bin 5 | 0.202095–0.308637 | 107 | 0.2659 |
| C7 | bin 6 | 0.308637–0.471346 | 165 | 0.3892 |
| C8 | bin 7 | 0.471346–0.719834 | 332 | 0.5993 |
| C9 | bin 8 | 0.719834–1.099321 | 515 | 0.9065 |
| C10 | bin 9 | 1.099321–1.678870 | 637 | 1.3520 |
| C11 | bin 10 | 1.678870–2.563949 | 392 | 1.9126 |
| C12 | upper tail | 2.563949–3.136165 | 24 | 2.6784 |

**6. Interpretation** — equal-width bins in ln(Sj*) because Sj* spans ~4
decades; tails kept as separate classes (nothing filtered, extremes get
their own JPS/curves).

**7. Say**: « 10 bins de largeur égale en ln(Sj*) entre P1 et P99, plus
deux classes de queue — rien n'est filtré. Le représentant de chaque
classe est la médiane des blocs réels — choix KCO*, configurable. »

---

## E. Count frequency vs volume weights

**1. Purpose** — two different statistics, do not confuse them.

**2/3. Code & inputs**
- **Count weight** (histogram + KCO* "count" reference): `w_i^N = N_i/N_tot`,
  set at class construction (`kco_star_model.py:604`: `weight=n_b/n`).
- **Volume weight** (KCO* active, professor's request):
  `kco_star_model.py:1259-1272`: `class_vol_sums[i] = Σ_{j∈i} Sj*_j³`
  (= Σ V_j); `weights_volume = class_vol_sums / total_vol`,
  `total_vol = 5854.8 m³`; asserted to sum to 1. Equation:
  `w_i^V = Σ_{j∈i} V_j / Σ_j V_j`.
- `weight_method="volume"` (`run_kco_varenne_demo.py:242`) selects w^V
  for `P_KCO*`; both weights are stored on every class.

**4. Calculation** — class C7: w^N = 165/2303 = 0.0716; w^V =
10.3878/5854.8 = 0.001774. Tail C12: w^N = 0.0104 but w^V =
495.8/5854.8 = **0.0847** — 24 blocks carry 8.5 % of the mass.

**5. Output** — `weight` (active), `weight_count`, `weight_volume`,
`volume_sum_m3` per class in the table; KCO* computed twice (volume =
primary; count = reference column in the D-table).

**6. Interpretation** — the histogram answers "fraction of BLOCKS per
bin"; KCO* answers "fraction of blasted ROCK VOLUME per structural
class" — volume weighting is the physically consistent choice for a
gradation (% passing by mass/volume).

**7. Say**: « L'histogramme est en part de blocs, mais KCO* prédit un
% passant — une fraction de volume : chaque classe est pondérée par
la somme des volumes de ses blocs, w_i = ΣV_j/ΣV_tot, comme demandé
par le Prof. La courbe count-weighted reste en référence. »

---

## F. Sj* → JPS

**1. Purpose** — convert each class's Sj*_rep to the Cunningham JPS
rating (drives JF → A → X50).

**3. Code** — `kco_model.py:308` `jps_from_joint_spacing(Sj, B, S)`;
reference length `sqrt(B·S)`; upper bound `0.95·sqrt(B·S)`.

**4. Calculation** (`:355-368`):
```
Sj < 0.1                      -> JPS 10
0.1 <= Sj < 0.3               -> JPS 20
0.3 <= Sj < 0.95·sqrt(BS)     -> JPS 80
Sj > sqrt(BS)                 -> JPS 50
0.95·sqrt(BS) <= Sj <= sqrt(BS) -> UNDEFINED -> ValueError (not guessed)
```
Varenne: `sqrt(BS) = sqrt(13.94) = 3.7336 m`; `0.95·sqrt(BS) = 3.5469 m`.

**5. Output** — C1–C3 (0.025–0.077) → **JPS 10**; C4–C6 (0.107–0.266) →
**JPS 20**; C7–C12 (0.389–2.678) → **JPS 80**. No class hits JPS 50 or
the undefined band.

**6. Interpretation** — small-block classes = closely jointed (easy to
break); large-block classes = widely jointed relative to the pattern.


---

## G. Classical KCO inputs

All in `BlastDesign(...)` at `run_kco_varenne_demo.py:144-167`:

| Input | Variable | Value | Unit | Provenance |
|---|---|---|---|---|
| B | `burden_m` | 3.4 | m | real design (user) |
| S | `spacing_m` | 4.1 | m | real design |
| H | `bench_height_m` | 14.0 | m | real design |
| Hole D | `hole_diameter_mm` | 114 | mm | real |
| Subdrill | `subdrill_m` | 0.0 | m | real |
| Charge L | `total_charge_m` | 14.0 | m | real (full column) |
| Q | `charge_per_hole_kg` | 165 | kg | real |
| q | `powder_factor_reported_kg_m3` | 0.8 | kg/m³ | reported; code cross-checks Q/(BSH) = 0.8455 — mode "reported" → **0.8 used** |
| RWS | `s_anfo_pct` | 100 | % | **ESTIMATE** — XL900 RWS unconfirmed (web gave 0.77–0.94, not used) |
| ρ | `rock_density_kg_m3` | 2700 | kg/m³ | professor |
| UCS | `ucs_mpa` | 200 | MPa | professor |
| E | `youngs_modulus_gpa` | 60 | GPa | professor |
| Rock mass | `rock_mass_case` | "jointed" | — | justified by DFN |
| JPA | `jpa_case` | "strike_perpendicular_to_face" | — | **ESTIMATE** → JPA 30 (`jpa_mapping_version="cunningham_2005"`) |
| JCF | `jf_includes_jcf` | False | — | baseline JF = JPS+JPA |
| Sj | `SJ_CLASSICAL_PROFESSOR_M` | 1.5 | m | **professor-prescribed**, overrides the SPACING mean 0.7532 m (`main.py:99`, `outputs/SPACING/raw_spacing_values.xlsx`, col `spacing`, region VARENNE) |
| W | `drill_accuracy_sd_m` | 0.3 | m | **ESTIMATE** |
| n_s | `timing_scatter_factor_ns` | 1.0 | — | Cunningham neutral |
| g(n) | `shift_factor_mode` | "no_shift" | — | g(n)=1 |
| A_T | `timing_factor` | 1.0 | — | disabled |
| Xmax | `xmax_vols` | same 2303 blocks | — | P95(Sj*) = 2.16225 m → Xmax = min(2.16225, B, S) = **2.16225 m** (in-situ governed) |

**Old/outdated values to flag**: `_C0 = 150 MPa` (crushed-zone UCS,
`run_kco_varenne_demo.py:319`) vs design `ucs_mpa = 200` — **à
confirmer**; `E = 60 GPa` reused as Esen E_d — provisional; the 50 m
blockometry (`block_shape_data_VARENNE.csv`) is only printed, not used;
reported q = 0.8 vs calculated 0.8455 (~5 % diff, tolerance 10 %).

---

## H. Rock factor (line by line)

**3. Code** — `predict_kco` (`kco_model.py:1696-1725`):
`jps_from_joint_spacing` → `joint_factor` → `rock_mass_description` →
`rock_density_influence` → `hardness_factor` → `blastability_index` →
`rock_factor_A`.

**4. Varenne numbers (JPS-80 = classical)**:
```
JPS = 80; JPA = 30 → JF = 110      (baseline, no JCF)
RMD = JF = 110                     ("jointed" massif)
RDI = 0.025·2700 − 50 = 17.5
HF  = UCS/5 = 200/5 = 40           (E = 60 ≥ 50 GPa → UCS branch)
BI  = 110 + 17.5 + 40 = 167.5
A   = 0.06·167.5 = 10.05
```
JPS-10 classes: JF 40, BI 97.5, A 5.85. JPS-20: JF 50, BI 107.5, A 6.45.

**7. Say**: « Chaîne publiée : JF = JPS+JPA, RMD = JF (jointé), RDI =
0,025ρ−50, HF = UCS/5 car E ≥ 50 GPa, BI = somme, A = 0,06·BI.
Avec UCS 200, ρ 2700 → A = 10,05 pour les classes JPS 80. »

---

## I. X50

**3. Code** — `kco_model.py:738` `x50_kuznetsov`; called at `:1776`
(classical) and `kco_star_model.py:1311` (per class).

**4.** `X50 = g(n)·A_T·A·Q^(1/6)·q^(−0.8)·(115/s_ANFO)^(19/30)` cm → ×10
mm. Mapping: g_n→g(n)=1, timing_factor→A_T=1, rock_factor_a→A,
charge_per_hole_kg→Q=165, powder_factor→q_used=0.8, s_anfo→RWS=100.

Numerical: 10.05×165^(1/6)×0.8^(−0.8)×1.15^0.6333
= 10.05×2.342×1.1954×1.0926 = **30.74 cm = 307.41 mm** (JPS-80 &
classical — identical because same JPS). JPS-10: 178.94 mm; JPS-20:
197.29 mm.

**Discrepancy note**: older files (suffix-free / `_volume_weighted` /
`_xmaxSBH`) contain obsolete UCS=100 or 50-m-DFN Xmax values; the
`_jps80_wipfines` suffix marks the current run.

**7. Say**: « X50 = A·Q^(1/6)·q^(−0,8)·(115/RWS)^(19/30), baseline
Ouchterlony 2005 sans shift ni timing : 307 mm pour les classes JPS 80.
Attention : RWS = 100 est un placeholder non confirmé. »

---

## J. Uniformity index n

**3. Code** — `kco_model.py:591` `uniformity_index_n` (Cunningham 2005 /
Ouchterlony & Sanchidrian 2019 eq. 48).

**4.** `n = n_s·√(2 − 30B/d)·√((1+S/B)/2)·(1 − W/B)·(L/H)^0.3·(A/6)^0.3`

JPS-80: 1×√(2−30·3.4/114)=1.0513; √((1+4.1/3.4)/2)=1.0502;
(1−0.3/3.4)=0.9118; (14/14)^0.3=1; (10.05/6)^0.3=1.1673
→ **n = 1.1752**. JPS-10: 0.9991; JPS-20: 1.0288 (n differs per class
because A differs).

**7. Say**: « n de Cunningham : géométrie (B, S, d), précision de forage
W = 0,3 m — placeholder — charge L/H, et le terme rocheux (A/6)^0,3.
C'est pourquoi n varie entre classes : A change avec JPS. »

---

## K. Swebrec b

**3. Code** — `kco_model.py:985` `b_parameter`:
`b = 2·ln2·n·ln(Xmax/X50)`.

**4.** JPS-80: 2×0.69315×1.17516×ln(2162.25/307.41) = **3.178**.
JPS-10: 3.451; JPS-20: 3.415. This is the Ouchterlony (2005) eq. (11c)
slope-matching-at-X50 form; the empirical `b = 0.5·x50^0.25·ln(...)`
alternative is **not** in the code (grep-verified).

**7. Say**: « b = 2 ln2 · n · ln(Xmax/X50), la forme d'égalisation des
pentes à X50 — pas la forme empirique en x50^0,25. »

---

## L. Swebrec cumulative curve

**3. Code** — `kco_model.py:1023` `swebrec_passing`; inverse
`swebrec_size_at_passing` (`:1065`) for percentiles. Grid:
`geomspace(1, xmax_mm, 500)` in `plot_main_comparison` / Excel export.

**4.** `P(x) = 100 / (1 + [ln(Xmax/x)/ln(Xmax/X50)]^b)`; P = 0 below,
100 at/above Xmax. Inverse: `X_P = Xmax·exp(−ln(Xmax/X50)·(100/P−1)^(1/b))`.

**5. Output** — P(x) arrays (% passing, mm grid); percentiles via the
closed-form inverse.

**7. Say**: « Chaque courbe Swebrec est évaluée sur une grille log de
500 points entre 1 mm et Xmax ; elle passe par (X50, 50 %) et vaut 100 %
à Xmax. »

---

## M. KCO* class calculations

**Per class** (`kco_star_model.py:1287-1362`): Sj*_rep → JPS → JF → RMD →
BI → A_i → n_i → X50_i → b_i. **Constant for all**: JPA 30, RDI 17.5,
HF 40, B/S/H/d/W/Q/q/RWS/n_s/g(n)/A_T, and Xmax = 2162.25 mm (computed
once, `:1274-1283`).

| class | Sj*_rep | JPS | A | X50 (mm) | w_i^V |
|---|---|---|---|---|---|
| C1 tail-low | 0.0249 | 10 | 5.85 | 178.94 | 7.6e-8 |
| C2 | 0.0511 | 10 | 5.85 | 178.94 | 2.7e-7 |
| C3 | 0.0770 | 10 | 5.85 | 178.94 | 1.8e-6 |
| C4 | 0.1073 | 20 | 6.45 | 197.29 | 5.4e-6 |
| C5 | 0.1698 | 20 | 6.45 | 197.29 | 3.8e-5 |
| C6 | 0.2659 | 20 | 6.45 | 197.29 | 3.5e-4 |
| C7 | 0.3892 | 80 | 10.05 | 307.41 | 0.00177 |
| C8 | 0.5993 | 80 | 10.05 | 307.41 | 0.0124 |
| C9 | 0.9065 | 80 | 10.05 | 307.41 | 0.0695 |
| C10 | 1.3520 | 80 | 10.05 | 307.41 | 0.2880 |
| C11 | 1.9126 | 80 | 10.05 | 307.41 | 0.5433 |
| C12 tail-up | 2.6784 | 80 | 10.05 | 307.41 | 0.0847 |

Class curves: `swebrec_passing(x, x50_i, 2162.25, b_i)`. Only **3 distinct
curves** exist (JPS takes 3 values): C1–C3, C4–C6, C7–C12 — the JPS-80
group carries w^V = 0.9996.

**7. Say**: « Chaque classe passe par la même chaîne KCO publiée ; seul
le Sj* représentatif varie → JPS → A → X50. Ici 12 classes = 3 courbes
distinctes parce que JPS ne prend que 3 valeurs. »

---

## N. Final KCO* weighted curve

**3. Code** — `kco_star_model.py:935` `kco_star_passing`; the loop does
`out += c.weight * swebrec_passing(x, c.x50_mm, c.xmax_mm, c.b)`
(`:967-968`); weights asserted to sum to 1. `weight_method="volume"`
→ `c.weight = w_i^V`.

**4.** `P_KCO*(x) = Σ_i w_i·P_i(x)` — mixture of Swebrec curves (KCO*
methodology, flagged as NOT a published equation in the docstring).
Percentiles via `kco_star_size_at_passing` (4000-pt log grid,
log-linear inversion): X20* 105.76, X50* 307.36, X80* 612.60,
X90* 813.84 mm.

**7. Say**: « La courbe finale est le mélange pondéré par volume des 12
courbes de classes : chaque classe contribue selon sa fraction du volume
rocheux. »

---

## O. Envelope

**3. Code** — `kco_star_model.py:972` `kco_star_class_envelope`:
`stack = vstack([P_i(x) for all classes]); return stack.min(0), stack.max(0)`.

**4.** `P_min(x) = min_i P_i(x)`, `P_max(x) = max_i P_i(x)`, pointwise
over the 12 (=3 distinct) class curves.

**6. Interpretation** — spread spanned by the class predictions
(structural variability). NOT a confidence interval — both docstring and
figure caption say so explicitly.

**7. Say**: « Le fuseau est l'enveloppe min–max des courbes de classes,
point par point — dispersion structurelle, pas un intervalle de
confiance. »

---

## P. In-situ DFN cumulative curve

**3. Code** — `kco_star_model.py:1042` `in_situ_passing_from_sj_star`:
```python
sj_sorted_mm = np.sort(sj_star_dist.sj_star_m) * 1000.0
passing_pct = 100.0 * (np.arange(1, n + 1) - 0.5) / n
```
Percentiles via `in_situ_size_at_passing` (`:1067`, log-linear).

**4/6.** Raw variable = **Sj* = V^(1/3)** in mm (equivalent-cube edge),
ALL 2303 pooled blocks, **count-based** empirical CDF with Hazen plotting
position (i−0.5)/n — no volume weighting. Values: D20 = 507.01,
D50 = 1024.43, D80 = 1620.71, D90 = 1897.36 mm. Pre-blast structure,
not a fragmentation prediction.

**7. Say**: « La courbe in-situ est la CDF empirique des 2303 Sj* en mm,
position de Hazen (i−0,5)/n — comptage pur, sans pondération de volume.
C'est la structure avant le tir. »

---

## Q. WipFrag

**2/3.** — `load_wipfrag_curve` (`run_kco_varenne_demo.py:107-136`).
File `assets/Résultats WIPFRAG.xlsx`, sheet `"2026-07-23"`, column
`"travaillée"` (`WIPFRAG_SHEET`, `WIPFRAG_CURVE` at `:103-104`). Loader
scans the sheet for the "travaillée" header cell; that column =
% passing, the next column = size (mm); numeric pairs sorted by size.
The 2026-05-11 sheet is NOT used (July blast = current production run).

**4.** Percentiles: `np.interp(p, wipfrag_passing, wipfrag_size_mm)` —
linear on measured points: D20 = 344.78, D50 = 540.75, D80 = 907.66,
D90 = 1645.53 mm.

**7. Say**: « La courbe mesurée vient de l'Excel WipFrag, feuille
2026-07-23, gradation 'travaillée' — colonne % passant + colonne taille
à côté, interpolation linéaire pour les percentiles. »

---

## R. Missing-fines correction (Prof. Aubertin)

**3. Code** — `run_kco_varenne_demo.py:316-353` + `_wipfrag_adjusted`
(`:345-350`).

**4. Verbatim**:
```
r_0 = 0.057 m (= D_h/2); ρ_e = 1200 kg/m³; D = 4500 m/s;
C_0 = 150 MPa; C_0,dyn = 3·C_0 = 450 MPa; H = 14 m
P   = ρ_e·D²/8 = 3037.5 MPa
r_c = r_0·√(P/C_0,dyn) = 0.057·√(6.75) = 0.14809 m
V_fines = 4·π·r_c²·H = 3.8583 m³       (4 holes, FULL cylinders)
V_block = S·B·H = 195.16 m³
P_fines = 100·V_fines/V_block = 1.977 %
```
Adjusted CDF: `P_adj(x) = P_fines + (100−P_fines)·P_orig(x)/100`,
anchor `P_adj(1 mm) = P_fines`.

**Assumptions to state plainly**: cylinder length = H = 14 m; FULL
cylinder volume (blasthole not subtracted — annulus variant
4π(r_c²−r_0²)H = 3.2867 m³ → 1.684 % computed but archived); factor 4 =
4 blastholes per cell (Prof. Aubertin's explicit prescription — à
confirmer if asked why 4). `_C0 = 150 MPa` here vs design
`ucs_mpa = 200` — flag if asked.

**7. Say**: « La correction du Prof. Aubertin ajoute les fines
manquantes dans WipFrag via 4 cylindres de zone concassée :
P = ρD²/8, r_c = r_0√(P/C_0,dyn) avec C_0,dyn = 3·UCS, V_f = 4πr_c²H
→ 1,98 % de fines, ajouté à 1 mm puis renormalisation de la courbe. »

---

## S. Esen et al. (2003) CZI

**3. Code** — `run_kco_varenne_demo.py` (Esen block, same file,
`_wipfrag_adjusted_esen` helper).

**4. Verbatim** (only difference from R = the r_c model):
```
E_d = 60 GPa (ASSUMPTION — input-Excel static E, dynamic unconfirmed)
ν_d = 0.25
P_b = ρ_e·D²/8 = 3037.5 MPa
K   = E_d/(1+ν_d) = 48000 MPa
σ_c = UCS = 150 MPa
CZI = P_b³/(K·σ_c²) = 25.9493
r_c = 0.812·r_0·CZI^0.219 = 0.812·0.057·2.0417 = 0.09443 m
r_c/r_0 = 1.6567
V_fines = 4π·r_c²·H = 1.5689 m³ → P_fines = 0.804 %
```
Then identical anchor + rescale as R. Results: adj D20 = 341.00,
D50 = 537.51, D80 = 904.78, D90 = 1639.55 mm (vs Aubertin 335.38 /
532.68 / 900.49 / 1630.65). Comparison figure `..._czmethods.png`;
comparison CSV `VARENNE_crushed_zone_methods_comparison...csv`.

**7. Say**: « Esen 2003 est en parallèle : même chaîne de correction,
seul r_c change — CZI = P_b³/(K·σ_c²) avec K = E_d/(1+ν_d), puis
r_c = 0,812 r_0 CZI^0,219. E_d = 60 GPa est une hypothèse provisoire —
le module dynamique n'est pas confirmé. »

---

## T. Figures and their sources

| Figure (file) | Produced by | What it shows |
|---|---|---|
| `VARENNE_sj_star_histogram_JPS{suffix}.png` | `run_kco_varenne_demo.py:931-974` | Sj* count histogram (60 geom bins) + KDE + JPS-coloured classes + P1/P99 |
| `VARENNE_sj_star_histogram_JPS_volume_share{suffix}.png` | `:976-1029` | Same but % of total volume per bin (comparison only) |
| `VARENNE_sj_star_P1_P99_tails.png` | `kco_star_model.py` tails fig | P1/P99 tail zoom |
| `VARENNE_kco_vs_kco_star_comparison{...suffix}.png` | `plot_main_comparison` (`kco_star_model.py`) | In-situ DFN + classical KCO + KCO* + WipFrag (+class curves/envelope options) |
| `VARENNE_kco_vs_kco_star_comparison_volume_weighted_xmaxSBH_jps80_wipfines_final.png` | same + fines | FINAL presentation fig: in-situ, classical KCO, KCO*, WipFrag orig, WipFrag adjusted (Aubertin) |
| `..._czmethods.png` | same + Esen | adds Esen CZI adjusted WipFrag |
| `VARENNE_reference_in_situ_blocksize_distribution.png` | `plot_reference_blocksize_curve.py` | reference in-situ block-size plot (DFN + D-values markers) |
| `VARENNE_kco_figure_data_..._wipfines.xlsx` | `run_kco_varenne_demo.py` Excel export | all curves as Excel sheets + native charts |
| `VARENNE_sj_star_log_bins_table{suffix}.csv` | `:689-692` | 12-class table (this doc, sect. D/M) |
| `VARENNE_D20_D50_D80_D90_table{suffix}.csv` | results summary | percentile table across methods |
| `VARENNE_kco_results_summary{suffix}.md` | `:708+` | code-generated audit summary — source of the numbers in this doc |
| `VARENNE_kco_star_weighting_comparison{suffix}.csv` | `:703-706` | count vs volume-weighted KCO* D-values |
| `VARENNE_crushed_zone_methods_comparison{suffix}.csv` | Esen block | Aubertin vs Esen r_c, V_f, P_f, adj D-values |
| `archive_annulus_comparison/` | — | archived annulus variant figures |

Suffix `_volume_weighted_xmaxSBH_jps80_wipfines` = current run: volume
weights + Xmax from SxBxH DFN + JPS-80 classes + fines corrections.

---

## U. 30 anticipated questions (short French answers)

1. **Pourquoi 30 réalisations ?** — Choix provisoire ; la convergence n'a
   été testée qu'aux dimensions placeholder, pas encore à S×B×H réel.
2. **Les 2303 blocs viennent d'une seule réalisation ?** — Non : poolage
   des 30 réalisations, ~77 blocs en moyenne par réalisation.
3. **Pourquoi pooler ?** — Pour densifier l'échantillon statistique d'une
   même cellule ; hypothèse explicite, pas 30 cellules physiques.
4. **Les P32 viennent d'où ?** — De la calibration sur le grand domaine
   50×50×50 m (`P32_calibrated_summary.csv`), appliqués à la cellule.
5. **Pourquoi la cellule réelle et pas le grand domaine ?** — La taille
   des blocs in-situ dépend du volume considéré : on prend la géométrie
   réelle du tir pour Sj* et Xmax.
6. **C'est quoi Sj* exactement ?** — V^(1/3), le côté du cube équivalent
   du bloc — hypothèse KCO*, pas une équation publiée.
7. **Comment sont construites les classes ?** — 10 bins de largeur égale
   en ln(Sj*) entre P1 et P99, plus 2 classes de queue ; rien de filtré.
8. **Pourquoi P1/P99 ?** — Bornes raisonnables de discrétisation ; les
   queues restent comme classes explicites au lieu d'être jetées.
9. **Quel est le Sj* représentatif ?** — La médiane des blocs réels de
   la classe.
10. **Pourquoi pas la moyenne ?** — La médiane est robuste à l'asymétrie
    lognormale ; c'est configurable (`representative_method`).
11. **Poids comptage vs volume ?** — Histogramme : part de blocs. KCO* :
    part de volume ΣV_j/ΣV, car une granulométrie est une fraction de
    masse/volume.
12. **Combien vaut le poids de la queue haute ?** — 24 blocs = 1,04 % du
    comptage mais 8,5 % du volume.
13. **Comment passe-t-on de Sj* à JPS ?** — Seuils de Cunningham avec
    √(B·S) = 3,73 m : <0,1→10 ; 0,1–0,3→20 ; 0,3–3,55→80 ; >3,73→50.
14. **Pourquoi pas de JPS 50 ?** — Aucun Sj* représentatif ne dépasse
    √(B·S) ; la bande 0,95√(BS)–√(BS) est indéfinie et génère une erreur.
15. **Pourquoi seulement 3 courbes de classes ?** — JPS ne prend que 3
    valeurs (10, 20, 80) → A, n, X50, b identiques dans chaque groupe.
16. **D'où vient Sj = 1,5 m pour KCO classique ?** — Valeur prescrite
    par le Prof. ; le code charge aussi la moyenne SPACING 0,7532 m mais
    elle est écrasée par la consigne.
17. **RWS = 100, c'est sûr ?** — Non : placeholder. La RWS réelle de la
    XL900 n'est pas confirmée — point ouvert.
18. **q = 0,8 ou 0,8455 ?** — 0,8 rapporté est utilisé (mode "reported") ;
    le calculé Q/(BSH) = 0,8455 sert de vérification.
19. **Comment A est calculé ?** — JF = JPS+JPA ; RMD = JF (jointé) ;
    RDI = 0,025ρ−50 ; HF = UCS/5 (E≥50 GPa) ; BI = somme ; A = 0,06·BI.
20. **Formule de X50 ?** — A·Q^(1/6)·q^(−0,8)·(115/RWS)^(19/30),
    baseline Ouchterlony 2005, pas de g(n) ni A_T : 307 mm.
21. **Formule de n ?** — Cunningham : n_s·√(2−30B/d)·√((1+S/B)/2)
    ·(1−W/B)·(L/H)^0,3·(A/6)^0,3 ; W = 0,3 m est un placeholder.
22. **Formule de b ?** — b = 2 ln2 · n · ln(Xmax/X50), égalisation de
    pente à X50 ; pas la forme empirique x50^0,25.
23. **D'où vient Xmax ?** — min(P95(Sj*), B, S) = min(2,162, 3,4, 4,1) =
    2,162 m, gouverné par le bloc in-situ ; le grand domaine n'est pas
    utilisé.
24. **Comment KCO* combine les classes ?** — Mélange pondéré par volume
    Σ w_i^V·P_i(x) — méthodologie KCO*, pas une équation publiée.
25. **L'enveloppe, c'est un intervalle de confiance ?** — Non : min–max
    point par point des courbes de classes ; dispersion structurelle.
26. **La courbe in-situ est volume-pondérée ?** — Non : CDF empirique en
    comptage, position de Hazen (i−0,5)/n, Sj* en mm, 2303 blocs.
27. **WipFrag : quelle feuille ?** — "2026-07-23", gradation
    "travaillée" ; la feuille de mai n'est pas utilisée.
28. **Pourquoi corriger WipFrag ?** — Les fines <1 mm sont sous-mesurées
    au chantier ; on ajoute la zone concassée estimée.
29. **Différence Aubertin vs Esen ?** — Seul r_c change : Aubertin
    r_0√(P/C_0,dyn) → 1,98 % de fines ; Esen CZI r_0·0,812·CZI^0,219 →
    0,80 %. E_d = 60 GPa est une hypothèse provisoire.
30. **Points faibles assumés ?** — RWS, JPA, W, n_s, le nombre 30 de
    réalisations, C_0 = 150 vs UCS = 200, le statique-vs-dynamique de E_d.

---

## V. Cheat sheet (one screen)

```
INPUTS   B=3.4  S=4.1  H=14  d=114 mm  Q=165 kg  q=0.8 (reported)
         rho=2700  UCS=200  E=60 GPa  RWS=100!  JPA=30!  W=0.3!  n_s=1
         (! = estimate/placeholder)
DFN      30 seeds 3001-3030 -> cell 4.1x3.4x14 -> 2303 blocks pooled
Sj*      Sj* = V^(1/3);  P1=0.0372 m  P99=2.564 m
CLASSES  10 equal-width ln bins in [P1,P99] + 2 tails = 12; rep = median
JPS      sqrt(BS)=3.734; 0.95sqrt(BS)=3.547
         <0.1->10 | 0.1-0.3->20 | 0.3-3.55->80 | >3.73->50
         -> only JPS 10/20/80 present -> 3 distinct class curves
CHAIN    JF=JPS+JPA | RMD=JF | RDI=0.025rho-50 | HF=UCS/5
         BI=RMD+RDI+HF | A=0.06 BI
KCO      A=10.05 | n=1.1752 | X50=307.41 mm | b=3.178 | Xmax=2162.25 mm
         Xmax = min(P95 Sj*=2.16225, B, S) m — in-situ governed
KCO*     P(x)=sum w_i^V P_i(x);  w_i^V = sum V / 5854.8 m3
         X20*=105.8 X50*=307.4 X80*=612.6 X90*=813.8 mm
IN-SITU  count-CDF (i-0.5)/n: D20=507 D50=1024 D80=1621 D90=1897 mm
WIPFRAG  sheet 2026-07-23 "travaillee": D20=345 D50=541 D80=908 D90=1646
FINES    Aubertin: rc=0.148 m, Pf=1.98% | Esen CZI: rc=0.0944 m, Pf=0.80%
         adjusted: P_adj(1mm)=Pf;  P_adj=Pf+(100-Pf)P_orig/100
CAVEATS  RWS=100, JPA case, W=0.3, n_reals=30, C_0=150 vs UCS=200,
         E_d=60 GPa provisional — all flagged in code/output md
```

**End-to-end numbers to quote**: 2303 blocks -> Sj* CDF -> 12 classes
(3 distinct curves) -> classical KCO X50 = 307.4 mm, KCO* X50 = 307.4 mm,
WipFrag D50 = 540.8 mm measured -> +1.98 % fines (Aubertin) or +0.80 %
(Esen).

