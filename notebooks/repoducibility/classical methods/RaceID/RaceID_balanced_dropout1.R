library(RaceID)
library(igraph)

read_file <- function(counts_file, groups_file){
  counts <- as.matrix(read.csv(counts_file, row.names=1))
  groups <- as.factor(read.csv(groups_file, row.names=1)$x)
  cell_type <- levels(groups)
  label <- as.integer(groups)
  return(list(data_matrix = counts, cell_type = cell_type, cell_label = label))
}

datascale<-function(x){
  if(x>10000) {
    return(20)
  }else if(2000<x& x<=10000) {
    return(30)
  }else  {return(50)}
}

RaceID_cluster<-function(data,label){
  
  scale<-datascale(ncol(data))
  sc <- SCseq(data)
  sc <- filterdata(sc,mintotal = 1000)
  
  sc <- compdist(sc,metric="pearson")
  
  sc<-clustexp(sc,cln=(max(label)-min(label)+1),sat=FALSE,bootnr=scale,FUNcluster = "kmeans")
  nmi<-compare(as.numeric(sc@cluster$kpart),label,method="nmi")
  ari<-compare(as.numeric(sc@cluster$kpart),label,method="adjusted.rand")
  
  return(c(ari,nmi))  
} 

path = "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets"

dropouts <- c(1)

ari <- rep(0, 50)
dim(ari) <- c(5, 10)
nmi <- rep(0, 50)
dim(nmi) <- c(5, 10)
clusters <- seq(1, 5)
duplicates <- seq(1, 10)

for(cluster in clusters){
  for(dropout in dropouts){
    for(duplicate in duplicates){
      print((cluster - 1)*10 + duplicate)
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
      race_list = RaceID_cluster(datacount, cell_label)
      ari[cluster, duplicate] <- race_list[1]
      nmi[cluster, duplicate] <- race_list[2]
    }
  }
}

save(ari, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedARI_RaceID_balanced_dropout1.rda")
save(nmi, file="/Users/Healthy/Desktop/final/scripts and results/classical methods/savedNMI_RaceID_balanced_dropout1.rda")