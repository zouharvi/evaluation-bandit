import scipy.stats
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss occurred")

def pval(scores1, scores2) -> float:
    if len(scores1) < 2 or len(scores2) < 2:
        return 1.0
    return scipy.stats.ttest_ind(
        scores1, scores2,
        equal_var=False,
        alternative="less",
    ).pvalue


# import matplotlib.pyplot as plt