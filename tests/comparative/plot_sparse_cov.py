import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV
import matplotlib.pyplot as plt
from covariance_selection_algorithm import calculate, calculate_with_modification
import time

# #############################################################################
# Generate the data
n_samples = 60
n_features = 6
seed = 1
prng = np.random.RandomState(seed)
prec = make_sparse_spd_matrix(n_features, alpha=.98,
                              smallest_coef=.4,
                              largest_coef=.7,
                              random_state=prng)
cov = linalg.inv(prec)
d = np.sqrt(np.diag(cov))
cov /= d
cov /= d[:, np.newaxis]
prec *= d
prec *= d[:, np.newaxis]
X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
X -= X.mean(axis=0)
X /= X.std(axis=0)

# #############################################################################
# Estimate the covariance
emp_cov = np.dot(X.T, X) / n_samples

model = GraphicalLassoCV(cv=5)
model.fit(X)
cov_ = model.covariance_
prec_ = model.precision_

significant_level = 0.5

classic_demp_time = time.time()
demp_cov_ = calculate(emp_cov, significant_level)
demp_prec_ = linalg.inv(demp_cov_)
classic_demp_time = time.time() - classic_demp_time

modified_demp_time = time.time()
demp_mod_cov_ = calculate_with_modification(emp_cov, significant_level)
demp_mod_prec_ = linalg.inv(demp_mod_cov_)

modified_demp_time = time.time() - modified_demp_time

print("{} {} {} {} seed{}".format(classic_demp_time,
                                  modified_demp_time,
                                  significant_level,
                                  n_features,
                                  seed))

# #############################################################################
# Plot the results
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)

# plot the covariances
covs = [('Empirical', emp_cov), ('DempCov', demp_cov_),
        ('DempCovMod', demp_mod_cov_), ('True', cov)]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i + 1)
    plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)


# plot the precisions
precs = [('Empirical', linalg.inv(emp_cov)), ('Demp', demp_prec_),
         ('DempMod', demp_mod_prec_), ('True', prec)]
vmax = .9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i + 5)
    plt.imshow(np.ma.masked_equal(this_prec, 0),
               interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s precision' % name)
    if hasattr(ax, 'set_facecolor'):
        ax.set_facecolor('.7')
    else:
        ax.set_axis_bgcolor('.7')

plt.show()
