from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, GroupKFold, TimeSeriesSplit

validation_methods = {
    "kfold": KFold,
    "stratified_kfold": StratifiedKFold,
    "leave_one_out": LeaveOneOut,
    "group_kfold": GroupKFold,
    "time_series_split": TimeSeriesSplit
}