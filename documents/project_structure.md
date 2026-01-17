📁 Project Structure
├── .gitattributes
├── .gitignore
├── ClassificatoreNPS
│   ├── .dockerignore
│   ├── Applying Machine Learning Algorithms to New Psychoactive Substances Screening.ipynb
│   ├── Dockerfile
│   ├── NPSClassifier.ipynb
│   ├── __init__.py
│   ├── api_clf_nps.py
│   ├── datasetNPS
│   │   ├── 1-s2.0-S2468170923000358-mmc1.csv
│   │   ├── 1-s2.0-S2468170923000358-mmc2.txt
│   │   ├── 1-s2.0-S2468170923000358-mmc3.zip
│   │   └── PENTION_EI_Complete.csv
│   ├── dnn_model_structure.png
│   ├── model
│   │   ├── Claudio
│   │   │   ├── balanced_random_forest_brf.pkl
│   │   │   ├── balanced_random_forest_brf_hmp.pkl
│   │   │   ├── balanced_random_forest_rfe.pkl
│   │   │   ├── dnn_model_structure.png
│   │   │   ├── dnn_spectra_version.keras
│   │   │   └── scale_dnn.pkl
│   │   ├── balanced_random_forest_brf.pkl
│   │   ├── balanced_random_forest_brf_hmp.pkl
│   │   ├── balanced_random_forest_rfe.pkl
│   │   ├── dnn_spectra_version.keras
│   │   ├── scale_dnn.pkl
│   │   ├── temp_config.json
│   │   ├── xgb_nps_model.json
│   │   └── xgb_scaler.pkl
│   ├── plots_model
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   ├── requirements.txt
│   ├── service_clf_nps.py
│   └── train_xgboost_nps.py
├── CorrectionDispersion_PIML
│   ├── .dockerignore
│   ├── CNNDataset.py
│   ├── Dockerfile
│   ├── MCxM_PIML.py
│   ├── __init__.py
│   ├── api_correction_piml.py
│   ├── binary_map_gen.py
│   ├── binary_maps_data
│   │   ├── amsterdam_netherlands_bbox.npy
│   │   └── amsterdam_netherlands_metadata_bbox.json
│   ├── cnn_train_pipeline_piml.py
│   ├── dataset
│   │   ├── nps_simulated_dataset_gaussiano_2025-11-24_PIML.csv
│   │   ├── nps_simulated_dataset_gaussiano_2025-11-24_PIML_partial.csv
│   │   ├── nps_simulated_dataset_gaussiano_2025-11-24_PIML_processed.csv
│   │   └── real_dispersion
│   │       ├── sim_0_conc_real_2025-11-24.npy
│   │       ├── sim_100_conc_real_2025-11-24.npy
│   │       ├── sim_101_conc_real_2025-11-24.npy
│   │       │   ...
│   │       ├── sim_98_conc_real_2025-11-24.npy
│   │       ├── sim_99_conc_real_2025-11-24.npy
│   │       └── sim_9_conc_real_2025-11-24.npy
│   ├── dataset_preprocessing_piml.py
│   ├── loss_function_piml.py
│   ├── models
│   │   ├── mcxm_piml_model_best.pth
│   │   └── mcxm_piml_model_final.pth
│   ├── plots
│   │   ├── distribuzioni_variabili_categoriche.png
│   │   ├── distribuzioni_variabili_continue.png
│   │   ├── matrice_correlazione.png
│   │   └── training_curves_piml.png
│   ├── requirements.txt
│   ├── service_correction_piml.py
│   ├── service_train_piml.py
│   ├── simulation_gen.py
│   └── train_model.py
├── EmissionSourceLocalization_PIML
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── __init__.py
│   ├── api_source_localization_piml.py
│   ├── emission_source_piml.ipynb
│   ├── models
│   │   ├── emission_source_model_piml.pkl
│   │   └── scaler_piml.pkl
│   ├── requirements.txt
│   └── service_source_localization_piml.py
├── MLOps
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── SensorSim_M.py
│   ├── __init__.py
│   ├── api_ingestion.py
│   ├── forensic_logger.py
│   ├── mock_retrain.py
│   ├── monitoring_service.py
│   └── requirements.txt
├── PentionSystem_M
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── static
│   │   ├── index.html
│   │   ├── logo.png
│   │   ├── script.js
│   │   ├── style.css
│   │   └── van_icon.png
│   └── ui_pention_m.py
├── README.md
├── docker-compose.yml
├── documents
│   ├── architecture
│   │   ├── Data Flow Model DSR.drawio
│   │   └── PENTION-M_Architecture.drawio
│   ├── forensic_outputs
│   │   └── forensic_report_SIM_20260112_182710.pdf
│   ├── requirements
│   │   ├── PENTION proposal_MION_vFINAL.pdf
│   │   ├── PENTION_initial_requirements_elicitation.xls
│   │   └── Requirements Elicitations Interview.docx
│   ├── stakeholder_workshop_2025-11-04
│   │   ├── (Audio) 2025-11-04.m4a
│   │   ├── 2025-11-04 09-59-04.mkv
│   │   ├── PENTION Requirements Elicitation Form (V2)(1-3).xlsx
│   │   ├── PENTION Requirements Elicitation Form(1-5).xlsx
│   │   ├── PENTION_LEA_Workshop_DutchPolice_Meeting_Summary_2025-11-04.docx
│   │   ├── PENTION_LEA_Workshop_DutchPolice_Meeting_Summary_2025-11-04.pdf
│   │   ├── Workshop-requirement engineering.pptx
│   │   └── transcript.txt
│   └── thesis_latex
│       ├── bib.bib
│       ├── chapters
│       │   ├── 01_introduzione.tex
│       │   ├── 02_background.tex
│       │   ├── 03_metodologia.tex
│       │   ├── 04_architettura.tex
│       │   ├── 05_validazione.tex
│       │   ├── 06_limitazioni.tex
│       │   └── 07_conclusioni.tex
│       ├── frontespizio.tex
│       ├── img
│       │   ├── PDF1.png
│       │   ├── PDF2.png
│       │   ├── PENTION-M_Architecture.png
│       │   ├── UI0.png
│       │   ├── UI1.png
│       │   ├── UI2.png
│       │   ├── UI3.png
│       │   ├── calibration_curve.png
│       │   ├── confidence_histogram.png
│       │   ├── confusion_matrix_xgb.png
│       │   ├── distribuzioni_variabili_continue.png
│       │   ├── dsr_process_flow.png
│       │   ├── logo_unisa.png
│       │   ├── model_accuracies.png
│       │   ├── noise_robustness.png
│       │   ├── piml_feature_correlation.png
│       │   ├── rfe_feature_importance.png
│       │   ├── source_localization_error_distribution.png
│       │   ├── source_localization_map.png
│       │   └── source_localization_scatter_xy.png
│       ├── main.pdf
│       ├── main.tex
│       └── preambolo.tex
├── gaussianPuff
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── Sensor.py
│   ├── __init__.py
│   ├── api_gaussian.py
│   ├── config.py
│   ├── gaussianFunction.py
│   ├── gaussianModel.py
│   ├── plot_utils.py
│   ├── requirements.txt
│   ├── sigmaCalculation.py
│   └── test_dispersion.py
├── logs
│   ├── drift_baseline.json
│   ├── forensic
│   │   ├── bundle_20251208_180247_a8f87755.json
│   │   ├── bundle_20251208_180254_f016b3e0.json
│   │   ├── bundle_20251208_180301_68039736.json
│   │   │   ...
│   │   ├── bundle_20260117_162540_12e9f20b.json
│   │   ├── bundle_20260117_162554_955d03ee.json
│   │   └── bundle_20260117_162915_4abfa3c4.json
│   ├── ingestion_log.jsonl
│   ├── model_registry.json
│   ├── modelops_retrain_log.jsonl
│   └── monitoring_log.jsonl
├── osm_cache
│   └── amsterdam_drive.graphml
├── project_structure.txt
├── shared_config
│   └── config_geo.py
└── validation
    ├── Emission
    │   └── emission_piml_metrics.json
    ├── EndToEnd
    │   ├── end_to_end_validation.ipynb
    │   └── run_end_to_end_test.py
    ├── Forensic
    │   ├── Forensic_Validation.ipynb
    │   ├── forensic_validation.py
    │   ├── forensic_validation_results.csv
    │   ├── tamper_test.py
    │   ├── tampered_bundle.json
    │   └── verify_bundle.py
    ├── MLOps
    │   ├── MLOps_Validation_Analysis.ipynb
    │   ├── mlops_pipeline_diagram
    │   ├── mlops_pipeline_diagram.png
    │   ├── mlops_pipeline_stress_test.py
    │   └── mlops_stress_results.csv
    ├── NPS
    │   ├── NPS_Calibration_and_Robustness.ipynb
    │   ├── nps_calibration_robustness.py
    │   └── results_nps
    │       ├── calibration_curve.png
    │       ├── confidence_histogram.png
    │       ├── noise_robustness.csv
    │       ├── noise_robustness.png
    │       └── summary.json
    ├── PIML
    │   ├── PIML_Validation_All.ipynb
    │   ├── metrics.py
    │   ├── utils_piml.py
    │   └── validation_results_piml.csv
    ├── build_final_kpi_table.py
    └── final_kpi_summary.csv