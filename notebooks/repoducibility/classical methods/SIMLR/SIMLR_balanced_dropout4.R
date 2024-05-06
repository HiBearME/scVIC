library(SIMLR)
library(igraph)

read_file <- function(counts_file, groups_file){
  counts <- as.matrix(read.csv(counts_file, row.names=1))
  groups <- as.factor(read.csv(groups_file, row.names=1)$x)
  cell_type <- levels(groups)
  label <- as.integer(groups)
  return(list(data_matrix = counts, cell_type = cell_type, cell_label = label))
}

SIMLR_cluster_large = function(data, label){
  res_large_scale = SIMLR_Large_Scale(X = data, c = length(unique(label)),normalize = TRUE)
  nmi = compare(label, res_large_scale$y$cluster, method = "nmi")
  ari = compare(label, res_large_scale$y$cluster, method = "adjusted.rand")
  return(c(ari,nmi))
}

path = "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets"

dropouts <- c(4)

ari <- rep(0, 50)
dim(ari) <- c(5, 10)
nmi <- rep(0, 50)
dim(nmi) <- c(5, 10)
clusters <- seq(1, 5)
duplicates <- seq(1, 10)
for(cluster in clusters){
  for(dropout in dropouts){
    for(duplicate in duplicates){
      set.seed(1)
      path_ratios <- paste(path, cluster  + 2, sep = "/")
      path_dropout <- paste(path_ratios, paste("dropout", dropout, sep = ""), sep="/")
      counts_file <- paste(paste(path_dropout, "counts_duplicate", sep="/"), duplicate, ".csv", sep="")
      groups_file <- paste(paste(path_dropout, "groups_duplicate", sep="/"), duplicate, ".csv", sep="")
      datainfor = read_file(counts_file, groups_file)
      datacount = as.matrix(datainfor$data_matrix)
      cell_type = datainfor$cell_type
      cell_label = datainfor$cell_label
      rownames(datacount)<-seq(nrow(datacount))
      colnames(datacount)<-seq(ncol(datacount))
      simlr_list = SIMLR_cluster_large(datacount, cell_label)
      ari[cluster, duplicate] <- simlr_list[1]
      nmi[cluster, duplicate] <- simlr_list[2]
    }
  }
}

save(ari, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedARI_SIMLR_balanced_dropout4.rda")
save(nmi, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedNMI_SIMLR_balanced_dropout4.rda")