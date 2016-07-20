import numpy as np
from scipy.stats import multivariate_normal

events = []
topics = []
nodes =[]
times = []
A = []
base_intensity = []
alphas = [] # shape: n_nodes * n_topics
phi = []  # shape: n_events * doc length * n_topics
docs =[]  # shape: n_events * doc length * n_words

def fdelta(t):
    return (t<3) and 1 or 0;

n_e = len(events)
r = np.random.dirichlet([1]*n_e, n_e)
n_topics = 5
n_words = 500
n_nodes = 5
sigma2 = 4
I = np.eye(n_topics)

# update for r
for i in len(events):
    r[i, 0] = base_intensity[nodes[i]] * multivariate_normal.pdf(topics[i], alphas[nodes[i]], sigma2*I)
    for j in len(events):
        if j==i: pass
        r[i,j] = A[j,i] * multivariate_normal.pdf(topics[i], topics[j], sigma2*I)* fdelta(times[i] - times[j])


# update for alpha
for n in range(n_nodes):
    alphas[n] = np.sum(r[:,0]*topics*([nodes == n]).astype('float'))

# updates for beta
xt = np.sum(docs, axis=2)
beta = sum([np.sum(xt*phi[:,:,k]) for k in range(n_topics)])  # ???????

