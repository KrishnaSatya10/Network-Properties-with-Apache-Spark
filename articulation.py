import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

# import findspark
# findspark.init()

sc = SparkContext("local", "degree.py")
# sc.setCheckpointDir(dirName='./checkpoints')
sqlContext = SQLContext(sc)


def articulations(g, usegraphframe=False):
    # Get the starting count of connected components
    # YOUR CODE HERE
    num_connected_components = g.connectedComponents().select(
        "component").distinct().count()

    vertex_list = g.vertices.rdd.map(lambda x: x.id).collect()

    # Default version sparkifies the connected components process
    # and serializes node iteration.
    if usegraphframe:
        # Get vertex list for serial iteration
        # YOUR CODE HERE
        # For each vertex, generate a new graphframe missing that vertex
        # and calculate connected component count. Then append count to
        # the output
        # YOUR CODE HERE
        vertex_articulations = []
        for vertex in vertex_list:
            new_vertices = g.vertices.filter(g.vertices.id != vertex)
            new_edges = g.edges.filter(
                (g.edges.src != vertex) & (g.edges.dst != vertex))
            subgraph = GraphFrame(new_vertices, new_edges)

            num_connected_components_vertex = subgraph.connectedComponents().select(
                "component").distinct().count()

            vertex_articulations.append((
                vertex, 1 if num_connected_components_vertex > num_connected_components else 0))

            return sqlContext.createDataFrame(vertex_articulations, ['id', 'articulation'])
    # # Non-default version sparkifies node iteration and uses networkx
    # # for connected components count.
    else:
        # YOUR CODE HERE
        def get_connected_components_subgraph(netx_graph, n):
            sub_graph = deepcopy(netx_graph)  # Copy the networkx graph.
            sub_graph.remove_node(n)  # Remove the node
            return nx.number_connected_components(sub_graph)

        netx_graph = nx.Graph()
        netx_graph.add_nodes_from(g.vertices.rdd.map(lambda r: r.id).collect())
        netx_graph.add_edges_from(g.edges.rdd.map(
            lambda r: (r.src, r.dst)).collect())

        vertex_articulations = []
        for vertex in vertex_list:
            num_connected_components_vertex = get_connected_components_subgraph(
                netx_graph, vertex)

            vertex_articulations.append(
                (vertex, 1 if num_connected_components_vertex > num_connected_components else 0))

        return sqlContext.createDataFrame(vertex_articulations, ['id', 'articulation'])


# filename = sys.argv[1]
filename = "9_11_edgelist.txt"

lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs, ['src', 'dst'])
e = e.unionAll(e.selectExpr('src as dst', 'dst as src')
               ).distinct()  # Ensure undirectedness

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()

# Create graphframe from the vertices and edges.
g = GraphFrame(v, e)

# Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
print("---------------------------")

print("Writing distribution to file articulations_out.csv")
df.toPandas().to_csv("articulations_out.csv")

# Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
