import sys
import re
import math
from pyspark import SparkConf, SparkContext

def truncate_5(x):
    factor = 10 ** 5
    if x >= 0:
        return math.floor(x * factor) / factor
    else:
        return math.ceil(x * factor) / factor

def main():
    if len(sys.argv) != 2:
        print("Usage: bin/spark-submit hw3_1.py <path_to_graph_file>")
        sys.exit(1)
    input_path = sys.argv[1]
    # Spark Context setting 
    conf = SparkConf().setAppName("PageRank_HW3") \
                      .set("spark.default.parallelism", "8") \
                      .set("spark.shuffle.compress", "false")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    #data parsing 
    def parse_line(line):
        try:
            line = line.strip()
            if not line or line.startswith("#"):
                return []
            parts = re.split(r'\s+', line)
            if len(parts) < 2:
                return []
            return [(int(parts[0]), int(parts[1]))]
        except ValueError:
            return []

    lines = sc.textFile(input_path)
    # remove duplicated edge
    links_rdd = lines.flatMap(parse_line).distinct().partitionBy(8).cache()
    
    # All node set 
    all_nodes = links_rdd.flatMap(lambda x: x).distinct()
    N = all_nodes.count()

    if N == 0:
        print("0")
        sc.stop()
        sys.exit(0)
    # count the number of outdegrees from each node 
    out_degrees = links_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y)
    
    # calculate deadend
    dead_ends_rdd = all_nodes.subtract(out_degrees.keys())
    count_dead_ends = dead_ends_rdd.count()

    # Spider Traps 
    # select the node that has only one outlink
    degree_one_nodes = out_degrees.filter(lambda x: x[1] == 1).keys()
    
    candidates_rdd = links_rdd.join(degree_one_nodes.map(lambda x: (x, None))) \
                              .map(lambda x: (x[0], x[1][0])) \
                              .cache()
    # Self-loops (A -> A)
    self_loops = candidates_rdd.filter(lambda x: x[0] == x[1]).map(lambda x: x[0])
    # Mutual-loops (A <-> B)
    # except Self-loop 
    pair_candidates = candidates_rdd.filter(lambda x: x[0] != x[1])
    # rdd_forward: (A, B) -> Key: A
    # rdd_check: (B, A) 
    rdd_forward = pair_candidates
    rdd_check = pair_candidates.map(lambda x: (x[1], x[0]))
    
    # if join operation is success, those nodes form a mutual loop
    mutual_loops = rdd_forward.join(rdd_check) \
                              .filter(lambda x: x[1][0] == x[1][1]) \
                              .map(lambda x: x[0])
    # union and sort 
    all_spider_traps = self_loops.union(mutual_loops).distinct().sortBy(lambda x: x)
    spider_traps_list = all_spider_traps.collect()

    # PageRank algorithm
    adj_list = links_rdd.groupByKey().mapValues(list).cache()
    ranks = all_nodes.map(lambda node: (node, 1.0 / N)).partitionBy(8)

    beta = 0.9
    max_iter = 50
    tolerance = 10**-5

    for i in range(max_iter):
        prev_ranks = ranks
        
        # calculate contibution 
        contribs = adj_list.join(ranks).flatMap(
            lambda x: [(dest, x[1][1] * beta / len(x[1][0])) for dest in x[1][0]]
        )
        
        # Leakage (Dead End + Teleportation) 
        total_mass = contribs.values().sum()
        missing_mass = (1.0 - total_mass) / N
        # rank update 
        ranks = contribs.reduceByKey(lambda x, y: x + y) \
                        .rightOuterJoin(all_nodes.map(lambda x: (x, 0.0))) \
                        .mapValues(lambda x: (x[0] if x[0] is not None else 0.0) + missing_mass)
        
        # convergence check
        diff = ranks.join(prev_ranks).map(lambda x: (x[1][0] - x[1][1])**2).sum()
        
        if math.sqrt(diff) < tolerance:
            break
    # Print the result 
    top_10 = ranks.takeOrdered(10, key=lambda x: (-x[1], x[0]))

    # (1) Dead Ends
    print(count_dead_ends)

    # (2) Spider Traps
    for spider_node in spider_traps_list:
        print(spider_node)

    # (3) Top-10 PageRank
    for node, score in top_10:
        print("{}\t{:.5f}".format(node, truncate_5(score)))
    sc.stop()

if __name__ == "__main__":
    main()