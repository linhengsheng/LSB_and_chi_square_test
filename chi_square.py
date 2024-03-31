import numpy as np
from PIL import Image
from scipy.special import gammainc

# chi2_distribution
"""
    if n variables independent to each other obey gaussian distribution,
        then: these n variables' square values obey a new distribution,
        which called Chi-square distribution

    density function:
        f = [(1/2)^(df/2) / Γ(df/2)] * [x^(df/2-1) * e^(-x/2)],
        where Γ(x) = ∫0->∞(t^(x-1)*e^(-t))dt
    df: degree of freedom
        represent number of independent variables
    when df = k, mean(f) = k, var(f) = 2k
"""

def chi_2_cdf(x, dof):
    """
        P(X<=x)=F_k(x)=γ(k/2, x/2) / Γ(k/2)
    :param x: chi_static
    :param dof: degree of freedom
    :return: P(X<x)
    """
    # gamma-inc(a, x) = γ(a, x) / Γ(a), 正则化的gamma函数(regular Γ)
    return gammainc(dof / 2, x / 2)


def chi_2_independence_test(observed):
    """
    :param: observed table
        p-value:
            -get statistic and freedom degree \n
            -return significance
    :return: chi2 statistic, p-value, freedom degree, expected
    """
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be positive.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    # calculate the expected frequency
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    total = np.sum(observed)
    """
        from the observed table's frequency, calculate the expected table:
        |    |   x1   |   x2  | 
        | y1 |total*p_col*prow| p_col_1 = row_total / total
        | y2 |                |
        |col_total/total| total

        eg: array[[10, 10, 20], [20, 20, 20]]
        -> expected :
            array[[12, 12, 16], [18, 18, 24]]
    """
    # np.outer: count vector a × vector b to a new matrix, row_i = np.dot(a_i, vector b)
    expected = np.outer(row_totals, col_totals) / total
    # calculate chi-square statistic
    chi_2 = np.sum((observed - expected) ** 2 / expected)
    # calculate freedom degree
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    # calculate p-value, 即计算累积分布积分(概率)，见‘latex->多媒体安全->实验一’
    p_value = 1 - chi_2_cdf(chi_2, dof)
    return chi_2, p_value, dof, expected


def chi_2_goodness_of_fit_test(observed: list | np.ndarray, expected: list | np.ndarray):
    """
    :param observed: frequency of each category, each category obey gaussian distribution
    :param expected: expected value of each category(E(x) of each category)
    :return: chi_2_static, p-value represent probability of chi_2_stat obeying chi_k^2
    """
    if isinstance(observed, list):
        observed = np.array(observed)
    if isinstance(expected, list):
        expected = np.array(expected)
    sum_observed = observed.sum()
    sum_expected = expected.sum()
    if sum_expected != sum_observed:
        expected = np.array([sum_observed * (e / sum_expected) for e in expected])
    chi_2_stat = np.sum((observed - expected) ** 2 / expected)
    """
        chi_square_test need to compare differece of observed values and expected value,
        then:
            need degree of freedom = k of chi_k^2 - 1 
    """
    p = chi_2_cdf(chi_2_stat, observed.shape[0]-1)
    return chi_2_stat, 1 - p


def chi2_analyze_image(image_path):
    """
        use chi_2_goodness_of_fit_test
    """
    image = Image.open(image_path)
    image_array = np.array(image.getdata()).flatten()
    """
        histogram statistic: \n
            -bins = int: evenly divide the interval
            -bins = [x, x, x]: self-defined interval
            -range: data's limits (min, max)
            return numbers of data located in each interval, delimiter of interval(numbers index)
    """
    observed, _ = np.histogram(image_array, bins=256)
    observed = observed.reshape(-1, 2)
    expected = np.mean(observed, axis=1)
    return chi_2_goodness_of_fit_test(observed[:, 0], expected)


if __name__ == '__main__':
    # , './secrets/miku.png' './images/miku.png'
    chi_2_stat, p = chi2_analyze_image('./secrets/miku.png')
    print("chi2_static:", chi_2_stat)
    print("p-value:", p)
    chi_2_stat, p = chi2_analyze_image('./images/miku.png')
    print("chi2_static:", chi_2_stat)
    print("p-value:", p)

