# Master Table Z — Magadi/Kajiado West meal feature matrix
### Z ∈ ℝ^(18 meals × 18 features) · each row is Z^k, the entry for meal k · matches `toy_minigrid_v3.py` and `meal_features_Z.csv`

**Physical / cost features:** kW draw, duration (steps of 15 min), kWh, electricity cost at flat KES 40 / green KES 30, charcoal kg and cost at KES 60/kg.
**Nutrition features:** kcal and macros (g) per serving.
**Decision features (new):** ingredient cost KES/serving at local prices · active prep labour (min) · taste/prestige (0–1) · tradition/ceremonial weight (0–1, Maasai context) · kid-acceptance (0–1) · batch/leftover potential (0–1) · fire_only flag (1 = zero clean-cooking proposition).
λ (agent taste weights) and γ (group offsets: house / school / kiosk) act on top of these columns; the model ships neutral defaults with per-agent noise.

| # | Meal (Z^k) | Type | Steps (min) | kW | kWh | Elec KES @40/@30 | Charc. kg / KES | kcal | P | C | F | Ingr. KES | Prep min | Taste | Tradit. | Kid | Batch | fire_only |
|---|-----------|------|------------|-----|-----|------------------|-----------------|------|----|----|----|-----------|----------|-------|---------|-----|-------|-----------|
| 1 | uji_honey_sweetpotato | ELEC | 3 (45) | 1.0 | 0.75 | 30 / 23 | 0.4 / 24 | 280 | 7 | 60 | 2 | 25 | 10 | 0.50 | 0.80 | 1.00 | 0.40 | 0 |
| 2 | mahamri_mbaazi_za_nazi | ELEC | 4 (60) | 1.2 | 1.20 | 48 / 36 | 0.6 / 36 | 780 | 21 | 112 | 27 | 60 | 40 | 0.80 | 0.40 | 0.90 | 0.80 | 0 |
| 3 | chai_roasted_sweetpotato_cassava | **FIRE** | 3 (45) | — | — | — | 0.6 / 36 | 360 | 6 | 70 | 6 | 30 | 10 | 0.55 | 0.85 | 0.80 | 0.30 | 1 |
| 4 | githeri_avocado (EPC) | ELEC | 4 (60) | 1.0 | 1.00 | 40 / 30 | 0.8 / 48 | 640 | 22 | 92 | 21 | 45 | 20 | 0.55 | 0.50 | 0.70 | 1.00 | 0 |
| 5 | ugali_ndengu_stew | ELEC | 4 (60) | 1.2 | 1.20 | 48 / 36 | 0.6 / 36 | 620 | 26 | 108 | 9 | 40 | 15 | 0.60 | 0.55 | 0.70 | 0.80 | 0 |
| 6 | ugali_sukuma_beef_stew | ELEC | 6 (90) | 1.5 | 2.25 | 90 / 68 | 0.8 / 48 | 800 | 37 | 98 | 27 | 90 | 30 | 0.80 | 0.60 | 0.80 | 0.60 | 0 |
| 7 | ugali_kuku_kienyeji_managu | ELEC | 6 (90) | 1.4 | 2.10 | 84 / 63 | 0.9 / 54 | 770 | 42 | 88 | 25 | 110 | 35 | 0.90 | 0.70 | 0.90 | 0.50 | 0 |
| 8 | chapati_maharagwe_ya_nazi | ELEC | 6 (90) | 1.6 | 2.40 | 96 / 72 | 0.9 / 54 | 680 | 21 | 92 | 25 | 55 | 45 | 0.85 | 0.50 | 0.95 | 0.70 | 0 |
| 9 | ugali_fried_tilapia_kachumbari | ELEC | 3 (45) | 1.5 | 1.13 | 45 / 34 | 0.5 / 30 | 730 | 46 | 74 | 28 | 100 | 20 | 0.85 | 0.50 | 0.60 | 0.20 | 0 |
| 10 | mukimo_beef_stew | ELEC | 5 (75) | 1.5 | 1.88 | 75 / 56 | 0.8 / 48 | 790 | 36 | 95 | 27 | 85 | 30 | 0.70 | 0.60 | 0.80 | 0.50 | 0 |
| 11 | matoke_beef | ELEC | 3 (45) | 1.2 | 0.90 | 36 / 27 | 0.5 / 30 | 420 | 19 | 58 | 13 | 60 | 20 | 0.60 | 0.50 | 0.70 | 0.50 | 0 |
| 12 | motori_bone_soup_cassava | ELEC | 10 (150) | 0.8 | 2.00 | 80 / 60 | 1.4 / 84 | 550 | 28 | 72 | 16 | 70 | 15 | 0.70 | 1.00 | 0.50 | 0.70 | 0 |
| 13 | nyama_choma_OVEN_kachumbari | ELEC | 4 (60) | 2.0 | 2.00 | 80 / 60 | 1.5 / 90 | 385 | 45 | 4 | 21 | 150 | 15 | 0.90 | 0.70 | 0.60 | 0.50 | 0 |
| 14 | nyama_choma_OPEN_FIRE → +16 | **FIRE** | 4 (60) | — | — | — | 1.5 / 90 | 385 | 45 | 4 | 21 | 150 | 15 | 1.00 | 1.00 | 0.60 | 0.50 | 1 |
| 15 | tilapia_catfish_GRILLED_FIRE → +17 | **FIRE** | 2 (30) | — | — | — | 0.7 / 42 | 280 | 40 | 2 | 12 | 90 | 15 | 0.80 | 0.50 | 0.50 | 0.20 | 1 |
| 16 | motori_bone_soup_cassava_FIRE | **FIRE** | 10 (150) | — | — | — | 1.8 / 108 | 550 | 28 | 72 | 16 | 70 | 15 | 0.70 | 1.00 | 0.50 | 0.70 | 1 |
| 17 | ugali_sukuma_side (w/ 14) | ELEC | 3 (45) | 1.5 | 1.13 | 45 / 34 | 0.5 / 30 | 410 | 11 | 77 | 7 | 30 | 15 | 0.60 | 0.60 | 0.80 | 0.30 | 0 |
| 18 | ugali_kachumbari_side (w/ 15) | ELEC | 2 (30) | 1.5 | 0.75 | 30 / 23 | 0.4 / 24 | 320 | 8 | 68 | 3 | 20 | 10 | 0.55 | 0.60 | 0.80 | 0.30 | 0 |

## Reading the decision columns

- **Ingredient cost** dominates day-to-day choice for low-income households (λ_ing_cost strongly negative): uji + sweet potato at KES 25 vs choma at KES 150/serving explains the frequency asymmetry before any energy pricing enters.
- **Prep labour** penalises chapati (45 min of kneading/rolling) and mahamri, which is why they cluster on weekends/feasts despite high taste scores — a kiosk (mama ntilie) agent with γ_prep more negative batches them instead.
- **Taste × tradition** is what keeps meals 14 and 16 on fire forever: choma scores 1.0/1.0, and no tariff touches it — that's the "0 proposition" made explicit as a feature rather than a hard-coded rule.
- **Kid + batch** are the school levers: γ_school upweights kid (0.9–1.0 meals: uji, chapati, kuku) and batch (githeri = 1.0, the canonical school lunch), and downweights taste — matching real Kenyan school-feeding menus.
- The λ/γ split works as intended: identical Z, different weights → houses pick beef plates, schools pick githeri, kiosks pick chapati and choma, without touching the meal definitions.

## Group offsets shipped as defaults (γ)

| Feature | house (base λ) | school γ | kiosk γ |
|---|---|---|---|
| taste | +1.0 | +0.3 | +1.4 |
| tradition | +0.6 | — | — |
| kid | +0.4 | +1.2 | — |
| batch | +0.3 | +1.5 | +1.0 |
| ing_cost | −0.8 | −1.5 | — |
| prep_min | −0.4 | — | −0.8 |
| kcal | +0.2 | — | — |

*All decision-column values are calibration placeholders — priced at rough 2026 local market rates and scored by judgement. Replace with survey data (KAOP or household interviews) when available; the Z structure and λ·z + γ machinery don't change.*
