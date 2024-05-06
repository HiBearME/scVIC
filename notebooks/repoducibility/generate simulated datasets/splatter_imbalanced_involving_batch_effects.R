library("splatter")
batch.prob <- (1 - 0.7) / (1 - 0.7 ** 3) * 0.7 ** c(0, 1, 2)
batch.prob <- batch.prob / sum(batch.prob)
ratios <- c(0.6, 0.7, 0.8, 0.9, 1.0)
dropout.mids <- c(-1.5, -1.0, -0.5)
duplicates <- seq(1, 10)
path = "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Imbalanced Datasets Involving Batch Effects"
if (!file.exists(path)){
dir.create(path)
}
for_ratio <- 0
for(ratio in ratios){
for_ratio <- for_ratio + 1
if (ratio==1){
group.prob = rep(1/5, 5)
}
else{
group.prob <- (1 - ratio) / (1 - ratio ** 5) * ratio ** c(0, 1, 2, 3, 4)
group.prob <- group.prob / sum(group.prob)
}
for_dropout <- 0
for(dropout.mid in dropout.mids){
for_dropout <- for_dropout + 1
seed <- 0
for(duplicate in duplicates){
seed <- seed + 1
batchCells <- round(2500 * batch.prob)
batchCells[3] = batchCells[3] + 2500 - sum(batchCells)
sim.groups <- splatSimulate(nGenes=2500,
group.prob = group.prob, method = "groups", batchCells=batchCells, dropout.mid=dropout.mid,
dropout.type = "experiment", dropout.shape = -1, de.facScale = 0.2, seed=seed)
counts <- assays(sim.groups)$counts
TrueCounts <- assays(sim.groups)$TrueCounts
groups <- colData(sim.groups)$Group
batches <- colData(sim.groups)$Batch
path_ratios <- paste(path, paste("ratio", for_ratio, sep = ""), sep = "/")
if (!file.exists(path_ratios)){
dir.create(path_ratios)
}
path_dropout <- paste(path_ratios, paste("dropout", for_dropout, sep = ""), sep="/")
if (!file.exists(path_dropout)){
dir.create(path_dropout)
}
write.csv(counts,  paste(paste(path_dropout, "counts_duplicate", sep="/"), duplicate, ".csv", sep=""))
write.csv(TrueCounts,  paste(paste(path_dropout, "TrueCounts_duplicate", sep="/"), duplicate, ".csv", sep=""))
write.csv(groups,  paste(paste(path_dropout, "groups_duplicate", sep="/"), duplicate, ".csv", sep=""))
write.csv(batches,  paste(paste(path_dropout, "batches_duplicate", sep="/"), duplicate, ".csv", sep=""))
DEFacGroup <- data.frame(DEFacGroup1 = rowData(sim.groups)[[paste0("DEFacGroup", 1)]])
for(i in 2:5)
{
DEFacGroup[[paste0("DEFacGroup", i)]] = rowData(sim.groups)[[paste0("DEFacGroup", i)]]
}
write.csv(DEFacGroup,  paste(paste(path_dropout, "DEFacGroup_duplicate", sep="/"), duplicate, ".csv", sep=""))
}
}
}
