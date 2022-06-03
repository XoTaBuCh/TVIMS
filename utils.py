import warnings

from scipy import stats
from statsmodels.distributions import ECDF

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math


def M(s):
    sum = 0.0
    for i in s:
        sum += i
    return sum / len(s)


def D(s):
    sum = 0.0
    for i in s:
        sum += i ** 2
    sum /= len(s)
    return (sum - M(s) ** 2)


def uniformF(x, a, b):
    if x < a:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return 1.0


def expF(x, lamb):
    if (x > 0):
        return 1.0 - math.exp(-1 * x * lamb)
    else:
        return 0.0


def normalF(x, m, lamb):
    return math.erf((x - m) / (lamb * (2 ** 0.5))) / 2


def geomF(x, P):
    return (1 - (1 - P) ** x)


def poissonF(x, m):
    return m ** x * math.exp(-m) / math.factorial(int(x))


def binomF(x, n, P):
    result = 0.0
    if x < 0:
        return 0

    for k in range(0, int(x)):
        result += P ** k * (1 - P) ** (n - k) * math.comb(n, k)

    return result


def plotUniform(a, b, n):
    uniform_arr = np.random.uniform(a, b, size=n)
    plt.hist(uniform_arr, density=True, bins=25, color="g", label="Uniform")
    plt.show()


def plotExponential(lamb, n):
    exponential_arr = np.random.exponential(scale=lamb, size=n)
    plt.hist(exponential_arr, density=True, bins=25, color="g", label="Exponential")
    plt.show()


def plotNormal(m, sigm, n):
    normal_arr = np.random.normal(m, sigm, n)
    plt.hist(normal_arr, density=True, bins=25, color="g", label="Normal")
    plt.show()


def plotGeom(P, n):
    geom_arr = np.random.geometric(p=P, size=n)
    geom_arr.sort()
    plt.hist(geom_arr, density=True, bins=25, color="g", label="Geometrical")
    plt.show()


def plotPoisson(M, n):
    poisson_arr = np.random.poisson(M, n)
    poisson_arr.sort()
    plt.hist(poisson_arr, density=True, bins=25, color="g", label="Poisson")
    plt.show()


def plotBinom(P, n):
    binom_arr = np.random.binomial(n, P, size=n)
    binom_arr.sort()
    plt.hist(binom_arr, density=True, bins=25, color="g", label="Binomial")
    plt.show()


def Kolmagorov(s):
    results = {}
    exp = M(s)
    var = D(s)
    n = len(s)
    print("Математическое ожидание - ", exp, " Дисперсия - ", var, "\n")

    betta = 0.95
    z = 1.96

    print("Доверительный интервал для M:")
    print(exp - z * (var / n) ** 0.5, ", ", exp + z * (var / n) ** 0.5)
    print("Доверительный интервал для D:")

    central_moment4 = sum([(i - exp) ** 4 for i in s]) / len(s)
    sigm = math.sqrt(central_moment4 / n - (n - 3) / (n * (n - 1)) * var ** 2)

    print(var - z * sigm, ", ", var + z * sigm)

    histogram(s)
    # histogramPandas(s)

    print("Равномерное распределение:")
    results["Равномерное"] = KolmagorovProcessing(s, stats.uniform.cdf, exp - ((3 * var) ** 0.5),
                                                  exp + ((3 * var) ** 0.5))
    plotUniform(exp - ((3 * var) ** 0.5), exp + ((3 * var) ** 0.5), n)

    print("Экспоненциальное распределение:")
    results["Экспоненциальное"] = KolmagorovProcessing(s, expF, 1 / exp)
    plotExponential(1 / exp, n)

    print("Нормальное распределение:")
    results["Нормальное"] = KolmagorovProcessing(s, stats.norm.cdf, exp, var ** 0.5)
    plotNormal(exp, var ** 0.5, n)

    try:
        print("Геометрическое распределение:")
        plotGeom(1 / exp, n)
        results["Геометрическое"] = KolmagorovProcessing(s, stats.geom.cdf, 1 / exp)

        print("Пуассоновское распределение:")
        results["Пуассоновское"] = KolmagorovProcessing(s, stats.poisson.cdf, exp)
        plotPoisson(exp, n)

        print("Биномиальное распределение:")
        results["Биномиальное"] = KolmagorovProcessing(s, stats.binom.cdf, n, exp / n)
        plotBinom(exp / n, n)
    except Exception as e:
        print()

    key_min = ""
    value_min = 100000000
    for key, value in results.items():
        if value[0] < value_min:
            key_min = key
            value_min = value[0]

    print("\nНаивероятнейшее распределение: для лямбда = ", value_min, " -- ",
          key_min, "\n")


def KolmagorovProcessing(s, func, *args):
    exp_func = ECDF(s)  # функция распределения

    experimental = []
    theoretical = []
    max_d = -1

    for x in np.linspace(0, max(s), len(s)):
        experimental_val = exp_func(x)
        theoretical_val = func(x, *args)

        experimental.append(experimental_val)
        theoretical.append(theoretical_val)
        max_d = max(max_d, abs(experimental_val - theoretical_val))

    p_value = stats.kstwo.sf(max_d, len(s))
    lambda_kol = np.sqrt(len(s)) * max_d

    plt.plot(np.linspace(0, max(s), len(s)), experimental, label="Empirical")
    plt.plot(np.linspace(0, max(s), len(s)), theoretical, label="Theoretical")
    plt.show()

    print("Lambda - ", lambda_kol)
    print("P-Value - ", p_value)

    return lambda_kol, p_value


def histogramPandas(s):
    plt.ylabel('P(x)')
    plt.xlabel('x')
    dataframe = pd.DataFrame(s)
    sns.distplot(dataframe, kde=True, bins=100)
    plt.show()


def histogram(s):
    plt.hist(s, density=True, bins=100)
    plt.ylabel('P(x)')
    plt.xlabel('x')
    plt.show()
