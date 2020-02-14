import networkx as nx
import pydot
import graphviz
from graphviz import Digraph
from networkx.drawing.nx_agraph import graphviz_layout

G=nx.Graph()
pos=nx.random_layout(G)

for i in range(0,20000):
    G.add_edge(z.iloc[i,4],z.iloc[i,1])
    #G.add_node(str(z.iloc[i,4]))
plt.figure(figsize=(300,300))
nx.draw(G,node_size=1200,alpha=0.6,edge_color='gray',node_color='green',with_labels=True,font_size=40)
plt.savefig("SOCIAL_NETWORK_emotions_no_edge2_EMO_NEG_OK_20K.png")
plt.show()
end=time.time()
print((end-start)/60)


########################################################################################

G=nx.read_edgelist("test.csv", delimiter=",")
pos=nx.spring_layout(G,k=200)

plt.figure(figsize=(300,300))
nx.draw_networkx_edges(G, pos, edge_color='blue', linewidths=0.25, arrows=True)

nx.draw_networkx_labels(G,pos,font_size=7,font_family='sans-serif',font_weight='bold')
#nx.draw_networkx_edge_labels(G,pos = graphviz_layout(G),z.iloc[:,0],label_pos=0.3)

nx.draw_networkx_nodes(G,pos,
                       nodelist=G.nodes(),node_size=300,alpha=0.6,
                       node_color='lightblue')

plt.savefig("SOCIAL_NETWORK_IMPROVED.png")
plt.show()

max(dict(G.degree()).items(), key = lambda x : x[1])


