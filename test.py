cluster_counts = {0: 4, 1: 7, 2: 2, 3: 5, 4: 15, 5: 3, 6: 12, 7: 612, 8: 1}

print(cluster_counts)

sorted_clusters = dict(sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True))
print(sorted_clusters)
mapping = {i: None for i in range(len(sorted_clusters))}
i = 0
for cluster in sorted_clusters.keys():
    mapping[i] = cluster
    i+=1
print(mapping)