"""Week 4 detection models and baselines (Proposal §6.4).

inputs     model inputs from label-free partitions: Site A windows reduced to segment
           means (or flattened), Site B window features; scaling fitted on training data
nets       the VAE (Site A) and dense autoencoder (Site B), built only from layers that
           Opacus supports, with training and reconstruction scoring (needs torch)
baselines  PCA reconstruction error and Isolation Forest (local only, scikit-learn)
"""
