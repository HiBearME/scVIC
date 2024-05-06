library(cidr)
library(igraph)

read_file <- function(counts_file, groups_file){
  counts <- as.matrix(read.csv(counts_file, row.names=1))
  groups <- as.factor(read.csv(groups_file, row.names=1)$x)
  cell_type <- levels(groups)
  label <- as.integer(groups)
  return(list(data_matrix = counts, cell_type = cell_type, cell_label = label))
}

CIDR_cluster = function(data, label){
  sc_cidr = scDataConstructor(data)
  sc_cidr = determineDropoutCandidates(sc_cidr)
  sc_cidr = wThreshold(sc_cidr)
  sc_cidr = scDissim(sc_cidr)
  sc_cidr = scPCA(sc_cidr,plotPC = FALSE)
  sc_cidr = nPC(sc_cidr)
  sc_cidr = scCluster(sc_cidr, nCluster = max(label) - min(label) + 1)
  nmi = compare(label, sc_cidr@clusters, method = "nmi")
  ari = compare(label, sc_cidr@clusters, method = "adjusted.rand")
  return(c(ari,nmi))
}

path = "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Imbalanced Datasets"

dropouts <- c(4)

ari <- rep(0, 50)
dim(ari) <- c(5, 10)
nmi <- rep(0, 50)
dim(nmi) <- c(5, 10)
ratios <- seq(1, 5)
duplicates <- seq(1, 10)
for(ratio in ratios){
  for(dropout in dropouts){
    for(duplicate in duplicates){
      set.seed(1)
      path_ratios <- paste(path, paste("ratio", ratio, sep = ""), sep = "/")
      path_dropout <- paste(path_ratios, paste("dropout", dropout, sep = ""), sep="/")
      counts_file <- paste(paste(path_dropout, "counts_duplicate", sep="/"), duplicate, ".csv", sep="")
      groups_file <- paste(paste(path_dropout, "groups_duplicate", sep="/"), duplicate, ".csv", sep="")
      datainfor = read_file(counts_file, groups_file)
      datacount = as.matrix(datainfor$data_matrix)
      cell_type = datainfor$cell_type
      cell_label = datainfor$cell_label
      rownames(datacount)<-seq(nrow(datacount))
      colnames(datacount)<-seq(ncol(datacount))
      cidr_list = CIDR_cluster(datacount, cell_label)
      ari[ratio, duplicate] <- cidr_list[1]
      nmi[ratio, duplicate] <- cidr_list[2]
    }
  }
}

save(ari, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedARI_CIDR_imbalanced_dropout4.rda")
save(nmi, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedNMI_CIDR_imbalanced_dropout4.rda")