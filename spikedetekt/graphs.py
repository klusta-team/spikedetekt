'''
Various simple routines for working with graph structures, used in defining the
spatial structure of the probes.
'''

def contig_segs(inds,padding=1):
    """ input: a list of indices. output: a list of slices representing contiguous blocks of indices,
    where padding is the distance that they can touch each other over
    e.g. contig_segs([0,1,2,4,5,6]) -> [ 0:3, 4:7] )"""
    segs_list = []
    i_start = 0
    n_inds = len(inds)
    while i_start < n_inds:
        i_end = i_start
        while i_end < n_inds-1:
            if inds[i_end+1] <= inds[i_end] + padding:
                i_end += 1
            else:
                break
        segs_list.append(range(inds[i_start],inds[i_end]+1))
        i_start = i_end + 1
    return segs_list

def complete_graph(n):
    return dict([(i,set(range(i)+range(i+1,n))) for i in range(n)])

def complete_if_none(MaybeGraph,n_nodes):
    if MaybeGraph is not None:
        return MaybeGraph
    else:
        return complete_graph(n_nodes)
    
def add_edge(G,src,targ):
    if src not in G: G[src] = set()
    if targ not in G: G[targ] = set()
    G[src].add(targ)
    G[targ].add(src)
    
def add_node(G,label):
    if label not in G: G[label] = set()
    
def edges(G):
    edge_list = []
    for src,targs in G.items():
        edge_list.extend([(src,targ) for targ in targs])
    return edge_list

def nodes(G):
    return G.keys()

def add_penumbra(mask, G, penumbra):
    '''
    Takes a channel mask, and adds penumbra nodes of neighbours in the graph G
    '''
    if penumbra==0:
        return mask
    newmask = mask.copy()
    for i, targets in G.iteritems():
        if mask[i]:
            for j in targets:
                newmask[j] = 1
    return add_penumbra(newmask, G, penumbra-1)
