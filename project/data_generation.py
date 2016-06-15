import numpy as np
from random import expovariate


from model import *

n_nodes = 5
window = 100
local_window = 3
mu = .02  # base intensity
A = np.array([[.3, .15, 0, 0, 0], [0, .3, .15, 0, 0], [0, 0, .3, .15, 0], [0, 0, 0, .3, .15], [.15, 0, 0, 0, .3]])
alphas = 2*np.eye(5)
sigma2 = 4
lambd = 200
n_topics = 5
n_words = 500
betas = np.random.dirichlet([1]*n_words, size=n_topics)

events = {}

def softmax(arr):
    arr = np.exp(np.array(arr))
    return arr/sum(arr)

def get_word(topic):
    return np.random.choice(np.arange(n_words), p=betas[topic])

def generate_event(base_time, rate, node, parent=None):
    l = []
    time = base_time + expovariate(rate)
    while time < window:
        if parent and time > base_time + local_window: break
        if parent:
            topic = np.random.multivariate_normal(parent.topic, sigma2*np.eye(alphas.shape[0]))
        else:
            topic = np.random.multivariate_normal(alphas[node], sigma2*np.eye(alphas.shape[0]))
        l.append(Event(node, time, parent, topic=topic))
        time += expovariate(rate)
    return l

for n in range(n_nodes):
    time = 0
    events[0] = generate_event(time, mu, n, parent=None)

layer = 1
time = 0

while len(events[layer-1]) > 0:
    events[layer] = []
    for parent in events[layer-1]:
        par_id  = parent.node
        for n in range(n_nodes):
            if A[par_id, n] == 0: continue
            events[layer] += generate_event(parent.time, A[par_id, n], n, parent)
    print ('%d finished' % layer)
    layer += 1


for l in events.values():
    for e in l:
        len = np.random.poisson(lambd)
        doc_topics = np.random.choice(np.arange(n_topics), len, p= softmax(e.topic))
        doc = np.array(map(get_word, doc_topics))
        e.doc = doc
        e.doc_topics = doc_topics

