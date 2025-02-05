from sklearn.preprocessing import KBinsDiscretizer

# Definir discretizadores
discretizers = {
    "k_bins": KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'),
    "quantile_bins": KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
}
