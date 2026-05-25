"""
proteomics_qc — portable quality-control pipeline for DIA proteomics data.

Designed as a teaching resource for proteomics courses. Operates on a
(samples × proteins) log2-intensity DataFrame produced by tools such as
Spectronaut, DIA-NN, or MaxQuant.

Subpackages
-----------
proteomics_qc.proteomics  — filtering, normalisation, outlier detection, batch QC
proteomics_qc.plots       — visualisation functions
proteomics_qc.report      — PDF report generation
"""
