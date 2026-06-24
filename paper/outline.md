# Paper Outline

*Section-by-section plan with claims and supporting results — filled in during Phase 7.*

## Working title

"Data Leakage in Fetal Health CTG Classification: A Reproducibility Audit and Corrected Benchmark"

## Sections

1. **Abstract** — gap, method, headline result (naive XX% → corrected XX%), checklist contribution
2. **Introduction** — CTG context, prevalence of leakage in published benchmarks, paper contributions
3. **Related Work** — Kapoor & Narayanan (2022), other leakage audits, CTG ML literature
4. **Dataset** — fetal-health dataset, class imbalance, why it tempts SMOTE misuse
5. **Methods** — L1–L5 leakage taxonomy, naive pipeline, corrected pipeline, evaluation protocol
6. **Experiments** — per-source leakage quantification, naive vs corrected table, calibration results
7. **Results** — headline accuracy gap, per-class breakdown, Pathological recall, calibration
8. **Discussion & Limitations** — one dataset, retrospective, scope limited to public benchmark CTG ML
9. **Checklist** — the CTG ML reporting checklist (cross-reference `checklist/`)
10. **Conclusion** — corrected benchmark as the reference point; checklist as the reusable artifact
