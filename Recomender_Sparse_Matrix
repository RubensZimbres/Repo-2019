from scipy.sparse import csr_matrix
import numpy as np

docs = [["hello", "friend", "hello"], ["goodbye", "old", "friend"]]
indptr = [0]
indices = []
data = []
termos=[]
vocabulary = {}
for d in docs:
    for term in d:
        termos.append(term)
        print(term)
        index = vocabulary.setdefault(term, len(vocabulary))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))

dataframe=csr_matrix((data, indices, indptr), dtype=int).toarray()

from collections import OrderedDict
columns=list(OrderedDict.fromkeys(termos))
