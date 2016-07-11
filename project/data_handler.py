from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

__author__ = 'mohsen'
import os
import sklearn
import numpy as np

names  = os.listdir('data/abstracts')
doc_ids = []
for id in names:
    n = id.split('.')[0]
    if n.startswith('11'):
        n = n[2:]
    if len(n) != 7:
        continue
    doc_ids.append(n)


edges = np.loadtxt('data/edges.txt', dtype='int')
d = {}
for i, e in enumerate(edges[:, 0]):
    dest = str(edges[i, 1])
    e = str(e)

    if e.startswith('11'):
        e = e[2:]
    if dest.startswith('11'):
        dest = dest[2:]
    if len(str(dest)) != 7 or len(str(e)) != 7:
        continue
    e = int(e)
    dest = int(dest)
    if e in d:
        d[e].append(dest)
    else:
        d[e] = [dest]


main_keys = []
for item, children in d.iteritems():
    if len(children) > 2:
        main_keys.append(item)

sim1_data = [d[i][0] for i in main_keys]
sim2_data = [d[i][1] for i in main_keys]
sim3_data = [d[i][2] for i in main_keys]

non_sim_data = []
alls = set(main_keys)
for k in main_keys:
    s = set(d[k])
    non_sim_data.append(alls.difference(s).pop())

cv = CountVectorizer('filename', stop_words='english')
count_vectors = cv.fit_transform(['data/abstracts/%s.abs' % id for id in main_keys])
tf_idf = TfidfTransformer(use_idf=True).fit_transform(count_vectors)

mapping = {}
for i, key in enumerate(main_keys):
    mapping[key] = i

sim1_data = [mapping[k] for k in sim1_data]
sim2_data = [mapping[k] for k in sim2_data]
sim3_data = [mapping[k] for k in sim3_data]
non_sim_data = [mapping[k] for k in non_sim_data]


def get_doc(id):
    with open('data/abstracts/%d.abs' % id) as f:
        lines = f.readlines()[6:]
    content = ' '.join(lines).replace('\\\\', '')
    return content
