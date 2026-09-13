"""
Bayesian A/B Testing (Beta-Binomial model)

Classic frequentist A/B tests give a single p-value.
Bayesian approach instead gives:
   - P(B > A)                 -> probability variant B is actually better
   - Expected uplift          -> how much better, on average
   - Credible interval        -> range of plausible uplift values

Model: each variant's conversion rate ~ Beta(alpha, beta)
       Beta is the conjugate prior for a Binomial likelihood,
       so posterior update is just alpha += conversions,
       beta += non-conversions. No MCMC needed.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def bayesian_ab_test(conversions_a, visitors_a, conversions_b, visitors_b,
                      prior_alpha=1, prior_beta=1, n_samples=100_000):

    # Posterior distributions (Beta-Binomial conjugacy)
    post_a = stats.beta(prior_alpha + conversions_a,
                         prior_beta + visitors_a - conversions_a)
    post_b = stats.beta(prior_alpha + conversions_b,
                         prior_beta + visitors_b - conversions_b)

    # Monte Carlo sampling from each posterior
    samples_a = post_a.rvs(n_samples)
    samples_b = post_b.rvs(n_samples)

    prob_b_better = np.mean(samples_b > samples_a)
    uplift = (samples_b - samples_a) / samples_a
    expected_uplift = np.mean(uplift)
    ci_low, ci_high = np.percentile(uplift, [2.5, 97.5])

    print("=" * 55)
    print(f"Variant A: {conversions_a}/{visitors_a} "
          f"({conversions_a/visitors_a:.2%} conversion)")
    print(f"Variant B: {conversions_b}/{visitors_b} "
          f"({conversions_b/visitors_b:.2%} conversion)")
    print("=" * 55)
    print(f"P(B > A)            : {prob_b_better:.2%}")
    print(f"Expected uplift (B)  : {expected_uplift:.2%}")
    print(f"95% credible interval: [{ci_low:.2%}, {ci_high:.2%}]")

    if prob_b_better > 0.95:
        print("Decision: Strong evidence B beats A")
    elif prob_b_better < 0.05:
        print("Decision: Strong evidence A beats B")
    else:
        print("Decision: Not enough evidence yet, keep testing")

    plot_posteriors(post_a, post_b)
    return prob_b_better, expected_uplift, (ci_low, ci_high)


def plot_posteriors(post_a, post_b):
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, post_a.pdf(x), label='Variant A', color='steelblue')
    plt.fill_between(x, post_a.pdf(x), alpha=0.3, color='steelblue')
    plt.plot(x, post_b.pdf(x), label='Variant B', color='darkorange')
    plt.fill_between(x, post_b.pdf(x), alpha=0.3, color='darkorange')
    plt.title('Posterior Conversion Rate Distributions')
    plt.xlabel('Conversion rate')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bayesian_ab_posteriors.png', dpi=120)
    plt.show()


if __name__ == "__main__":
    # Example: A = old design, B = new design
    conversions_a, visitors_a = 180, 2000   # 9.0% conversion
    conversions_b, visitors_b = 220, 2000   # 11.0% conversion

    bayesian_ab_test(conversions_a, visitors_a, conversions_b, visitors_b)
