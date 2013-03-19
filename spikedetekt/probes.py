'''
Construct a probe object from a .probe file with:

    probe = Probe(filename)

The file should have a form something like:

    probes = {
        1: [
            (0, 1), (0, 2),
            (1, 2), (1, 3),
            ...
            ],
        2: [...],
        ...
        }

The file is a Python file which should define a dictionary variable probes,
with keys the shank number, and values a list of channel pairs defining the
edges of the graph.

The Probe object has the following attributes:

    num_channels
        The number of channels used (1+the maximum channel number referred to)
    channel_graph
        A dictionary with keys the shank number, and values being graphs. A
        graph being a dictionary with keys the nodes (channel number) and
        values the set of all connected nodes. (So each channel in an edge is
        referred to twice in this data structure.)
    shanks_set
        The set of shank numbers
    channel_set
        A dictionary with keys the shank numbers, and values the set of channels
        for that shank
    channel_to_shank
        A dictionary with keys the channel numbers and values the corresponding
        shank number.
    probes
        The raw probes dictionary definition in the file
'''

__all__ = ['Probe']

class Probe(object):
    def __init__(self, filename):
        ns = {}
        probetext = open(filename, 'r').read()
        try:
            exec probetext in ns
        except Exception, e:
            raise IOError("Cannot parse probe file, error encountered: "+str(e))
        if 'probes' not in ns:
            raise IOError("Cannot parse probe file, no 'probes' dict found.")
        probes = ns['probes']
        self.probes = probes
        N = max(max(max(t) for t in edges) for edges in probes.itervalues())
        self.num_channels = N+1
        self.shanks_set = set(probes.keys())
        # sanity check on graphs, no repeated pairs or self-connections
        for edges in probes.itervalues():
            pairs = []
            for i, j in edges:
                if i<j:
                    pairs.append((i, j))
                elif i>j:
                    pairs.append((j, i))
                else:
                    raise ValueError("Probe graph doesn't allow self-connections.")
            if len(set(pairs))<len(pairs):
                raise ValueError("Repeated edge found in probe graph.")
        # construct the vertex set for each probe, check there are no overlaps
        self.channel_set = vs = {}
        used_vertices = set()
        for probenum, edges in probes.iteritems():
            curvs = set()
            for i, j in edges:
                curvs.add(i)
                curvs.add(j)
            if curvs.intersection(used_vertices):
                raise ValueError("Probe graphs overlap.")
            used_vertices.update(curvs)
            vs[probenum] = curvs
        # construct vertex->probe mapping
        self.channel_to_shank = v2p = {}
        for probenum in probes.keys():
            curvs = vs[probenum]
            for v in curvs:
                v2p[v] = probenum
        # construct unified graph
        self.channel_graph = G = {}
        for probenum, edges in probes.iteritems():
            for i, j in edges:
                if i in G:
                    ni = G[i]
                else:
                    ni = G[i] = set()
                if j in G:
                    nj = G[j]
                else:
                    nj = G[j] = set()
                ni.add(j)
                nj.add(i)
