from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import CategoricalNB

# Definir discretizadores
discretizers = {
    "k_bins": KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'),
    "categoricalNB" : CategoricalNB()
}
