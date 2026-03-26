import networkx as nx


def build_graph(relations):
    G = nx.DiGraph()

    seen = set()

    for rel in relations:
        subj = rel["subject"]
        obj = rel["object"]
        predicate = rel["relation"]

        key = (subj, obj, predicate)

        
        if key in seen:
            continue
        seen.add(key)

        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=predicate)

    return G