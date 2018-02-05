import sets
import sys
sys.path.append('.')
sys.path.append('./lib/python')
import PyBliss


def report(perm, text = None):
    print text, perm


def traverse1(G, N, R, stats):
    """
    Enumerate all graphs over N vertices up to isomorphism
    by using a form of the 'trivial' method.
    All graphs are generated and isomorph rejection is only applied
    to the 'leaf graph'.
    """
    if G.nof_vertices() == N:
        canform = G.relabel(G.canonical_labeling())
        if canform not in R:
            stats.nof_graphs += 1
            R.add(canform)
        return
    i = G.nof_vertices()
    v = "v"+str(i)
    for k in xrange(0,pow(2,i)):
        G.add_vertex(v)
        for j in xrange(0, i):
            if (k & 0x01) == 1:
                G.add_edge(v, "v"+str(j))
            k = k / 2
        traverse1(G, N, R, stats)
        G.del_vertex(v)


def traverse2(G, N, R, stats):
    """
    Enumerate all graphs over N vertices up to isomorphism
    by using a form of the 'folklore' method.
    Graphs are build iteratively and isomorph rejection is applied
    to all graphs.
    """
    canform = G.relabel(G.canonical_labeling())
    if canform in R:
        return
    R.add(canform)
    if G.nof_vertices() == N:
        stats.nof_graphs += 1
        return
    i = G.nof_vertices()
    v = "v"+str(i)
    for k in xrange(0,pow(2,i)):
        G.add_vertex(v)
        for j in xrange(0, i):
            if (k & 0x01) == 1:
                G.add_edge(v, "v"+str(j))
            k = k / 2
        traverse2(G, N, R, stats)
        G.del_vertex(v)


def traverse3(G, N, stats):
    """
    Enumerate all graphs over N vertices up to isomorphism
    by using a form the 'weak canonical augmentation' method.
    """
    if G.nof_vertices() == N:
        stats.nof_graphs += 1
        return
    i = G.nof_vertices()
    vertices = G.get_vertices()
    children = sets.Set()
    for k in xrange(0, pow(2, i)):
        G.add_vertex('tmp')
        for j in xrange(0, i):
            if (k & 0x01) == 1:
                G.add_edge('tmp', vertices[j])
            k = k / 2
        child = G.relabel(G.canonical_labeling())
        G.del_vertex('tmp') # restore G
        if child in children:
            continue
        child2 = child.copy()
        child2.del_vertex(0)
        child2_canform = child2.relabel(child2.canonical_labeling())
        if child2_canform.get_isomorphism(G) != None:
            children.add(child)
    for child in children:
        traverse3(child, N, stats)


class Stats:
    def __init__(self):
        self.nof_graphs = 0

G = PyBliss.Graph()
G.add_vertex('v1')
G.add_vertex('v2')
G.add_vertex('v3')
G.add_vertex('v4')
G.add_edge('v1','v2')
G.add_edge('v1','v3')
G.add_edge('v2','v3')
G.add_edge('v1','v4')
print "Computing generators for the automorphism group of the graph:"
G.write_dot(sys.stdout)
G.find_automorphisms(report, "Aut gen:")
canlab = G.canonical_labeling()
print "A canonical labeling of the graph is:",canlab
print "The canonical form of the graph is:"
G.relabel(canlab).write_dot(sys.stdout)

N = 3
stats = Stats()
G = PyBliss.Graph()
traverse1(G, N, sets.Set(), stats)
print "There are "+str(stats.nof_graphs)+" non-isomorphic graphs with "+str(N)+" vertices"

N = 5
stats = Stats()
G = PyBliss.Graph()
traverse2(G, N, sets.Set(), stats)
print "There are "+str(stats.nof_graphs)+" non-isomorphic graphs with "+str(N)+" vertices"

N = 6
stats = Stats()
G = PyBliss.Graph()
traverse3(G, N, stats)
print "There are "+str(stats.nof_graphs)+" non-isomorphic graphs with "+str(N)+" vertices"
