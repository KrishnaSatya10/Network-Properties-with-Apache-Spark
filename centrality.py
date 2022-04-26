from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from pyspark.sql.functions import explode

import findspark
findspark.init()

sc = SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)


def closeness(g):

    # Get list of vertices. We'll generate all the shortest paths at
    # once using this list.
    # YOUR CODE HERE
    paths_from_vertices = g.shortestPaths(
        landmarks=g.vertices.rdd.map(lambda x: x.id).collect())

    # first get all the path lengths.
    # print(paths_from_vertices.take(1)) -- Understand how the row looks like
    path_lengths = paths_from_vertices.select('id', explode('distances'))

    # Break up the map and group by ID for summing
    # Sum by ID
    distance_sums_by_id = path_lengths.selectExpr('id', 'value').groupBy(
        'id').agg(functions.sum('value').alias('distance_sums'))

    # Get the inverses and generate desired dataframe.
    closeness_by_id = distance_sums_by_id.selectExpr(
        'id', '1/distance_sums as closeness')

    return closeness_by_id


print("Reading in graph for problem 2.")
graph = sc.parallelize([('A', 'B'), ('A', 'C'), ('A', 'D'),
                        ('B', 'A'), ('B', 'C'), ('B', 'D'), ('B', 'E'),
                        ('C', 'A'), ('C', 'B'), ('C', 'D'), ('C', 'F'), ('C', 'H'),
                        ('D', 'A'), ('D', 'B'), ('D', 'C'), ('D',
                                                             'E'), ('D', 'F'), ('D', 'G'),
                        ('E', 'B'), ('E', 'D'), ('E', 'F'), ('E', 'G'),
                        ('F', 'C'), ('F', 'D'), ('F', 'E'), ('F', 'G'), ('F', 'H'),
                        ('G', 'D'), ('G', 'E'), ('G', 'F'),
                        ('H', 'C'), ('H', 'F'), ('H', 'I'),
                        ('I', 'H'), ('I', 'J'),
                        ('J', 'I')])

e = sqlContext.createDataFrame(graph, ['src', 'dst'])
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()
print("Generating GraphFrame.")
g = GraphFrame(v, e)

print("Calculating closeness.")
# closeness(g).sort('closeness', ascending=False).show()

print("Writing distribution to file centrality_out.csv")
closeness(g).sort('closeness', ascending=False).toPandas().to_csv(
    "centrality_out.csv")
