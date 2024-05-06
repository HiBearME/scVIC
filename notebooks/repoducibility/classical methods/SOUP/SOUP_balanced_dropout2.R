library(SOUP)
library(igraph)

read_file <- function(counts_file, groups_file){
  counts <- as.matrix(read.csv(counts_file, row.names=1))
  groups <- as.factor(read.csv(groups_file, row.names=1)$x)
  cell_type <- levels(groups)
  label <- as.integer(groups)
  return(list(data_matrix = counts, cell_type = cell_type, cell_label = label))
}

SOUP_cluster=function(data,label){
  data = t(data)
  colnames(data) = as.vector(seq(ncol(data)))
  data<-log2(scaleRowSums(data)*10^4 + 1)
  select.out = selectGenes(data, DESCEND = FALSE, type = "log")
  select.genes = as.vector(select.out$select.genes)
  data = data[, colnames(data) %in% select.genes]
  soup = SOUP(data, Ks = c(max(label) - min(label) + 1), type = "log")
  pred = as.vector(soup$major.labels[[1]])
  nmi=compare(label, pred, method = "nmi")
  ari=compare(label, pred, method = "adjusted.rand")
  return(c(ari,nmi))
}

path = "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets"

dropouts <- c(2)

ari <- rep(0, 50)
dim(ari) <- c(5, 10)
nmi <- rep(0, 50)
dim(nmi) <- c(5, 10)
clusters <- seq(1, 5)
duplicates <- seq(1, 10)
for(cluster in clusters){
  for(dropout in dropouts){
    for(duplicate in duplicates){
      print((cluster - 1) * 10 + duplicate)
      set.seed(1)
      path_ratios <- paste(path, cluster + 2, sep = "/")
      path_dropout <- paste(path_ratios, paste("dropout", dropout, sep = ""), sep="/")
      counts_file <- paste(paste(path_dropout, "counts_duplicate", sep="/"), duplicate, ".csv", sep="")
      groups_file <- paste(paste(path_dropout, "groups_duplicate", sep="/"), duplicate, ".csv", sep="")
      datainfor = read_file(counts_file, groups_file)
      datacount = as.matrix(datainfor$data_matrix)
      cell_type = datainfor$cell_type
      cell_label = datainfor$cell_label
      rownames(datacount)<-seq(nrow(datacount))
      colnames(datacount)<-seq(ncol(datacount))
      soup_list = SOUP_cluster(datacount, cell_label)
      ari[cluster, duplicate] <- soup_list[1]
      nmi[cluster, duplicate] <- soup_list[2]
    }
  }
}

save(ari, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedARI_SOUP_balanced_dropout2.rda")
save(nmi, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedNMI_SOUP_balanced_dropout2.rda")