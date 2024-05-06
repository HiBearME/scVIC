library("splatter")
batch.prob <- (1 - 0.7) / (1 - 0.7 ** 3) * 0.7 ** c(0, 1, 2)
batch.prob <- batch.prob / sum(batch.prob)
batchCells <- round(2500 * batch.prob)
batchCells[3] = batchCells[3] + 2500  - sum(batchCells)
n_clusters <- c(3, 4, 5, 6, 7)
dropout.mids <- c(-1.5, -1.0, -0.5, 0.0, 0.5)
duplicates <- seq(1, 10)
path = "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets Involving Batch Effects"
if (!file.exists(path)){
  dir.create(path)
}

for(n_cluster in n_clusters){
  for_dropout <- 0
  for(dropout.mid in dropout.mids){
    for_dropout <- for_dropout + 1
    seed <- 0
    for(duplicate in duplicates){
      seed <- seed + 1
      sim.groups <- splatSimulate(nGenes=2500,
        group.prob = rep(1/n_cluster, n_cluster), method = "groups", batchCells=batchCells, dropout.mid=dropout.mid,
        dropout.type = "experiment", dropout.shape = -1, de.facScale = 0.2, seed=seed)
      
      counts <- assays(sim.groups)$counts
      TrueCounts <- assays(sim.groups)$TrueCounts
      groups <- colData(sim.groups)$Group
      batches <- colData(sim.groups)$Batch
      path_clusters <- paste(path, n_cluster, sep = "/")
      if (!file.exists(path_clusters)){
        dir.create(path_clusters)
      }
      path_dropout <- paste(path_clusters, paste("dropout", for_dropout, sep = ""), sep="/")
      if (!file.exists(path_dropout)){
        dir.create(path_dropout)
      }
      
      write.csv(counts,  paste(paste(path_dropout, "counts_duplicate", sep="/"), duplicate, ".csv", sep=""))
      write.csv(TrueCounts,  paste(paste(path_dropout, "TrueCounts_duplicate", sep="/"), duplicate, ".csv", sep=""))
      write.csv(groups,  paste(paste(path_dropout, "groups_duplicate", sep="/"), duplicate, ".csv", sep=""))
      write.csv(batches,  paste(paste(path_dropout, "batches_duplicate", sep="/"), duplicate, ".csv", sep=""))
      DEFacGroup <- data.frame(DEFacGroup1 = rowData(sim.groups)[[paste0("DEFacGroup", 1)]])
      for(i in 2:n_cluster)
      {
        DEFacGroup[[paste0("DEFacGroup", i)]] = rowData(sim.groups)[[paste0("DEFacGroup", i)]]
      }
      write.csv(DEFacGroup,  paste(paste(path_dropout, "DEFacGroup_duplicate", sep="/"), duplicate, ".csv", sep=""))
    }
  }
}
