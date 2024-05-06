library(Seurat)

path = "/Users/Healthy/Desktop/final/scripts and results/generate simulated datasets/data/Imbalanced Datasets Involving Batch Effects"

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
      batches_file <- paste(paste(path_dropout, "batches_duplicate", sep="/"), duplicate, ".csv", sep="")
      rawdata <- read.csv(counts_file)
      # column is cell, row is gene
      rawdata <- rawdata[,2:2501] 
      batch <- read.csv(batches_file)
      batch <- batch[,2]
      group <- read.csv(groups_file)
      group <- group[,2]
      
      tt1 <- rawdata[batch=="Batch1"]
      tt2 <- rawdata[batch=="Batch2"] 
      tt3 <- rawdata[batch=="Batch3"]
      
      tt1 <- CreateSeuratObject(tt1)
      tt2 <- CreateSeuratObject(tt2)
      tt3 <- CreateSeuratObject(tt3)
      
      rawdata.list <- list(batch1=tt1, batch2=tt2, batch3=tt3)
      for (i in 1:length(rawdata.list)) {
        rawdata.list[[i]] <- NormalizeData(rawdata.list[[i]], verbose = FALSE)
        rawdata.list[[i]] <- FindVariableFeatures(rawdata.list[[i]], selection.method = "vst", nfeatures = 500,
                                                   verbose = FALSE)
      }
      
      features <- SelectIntegrationFeatures(object.list = rawdata.list, nfeatures=500)
      
      rawdata.anchors <- FindIntegrationAnchors(object.list = rawdata.list, anchor.features = features)
      
      rawdata.integrated <- IntegrateData(anchorset = rawdata.anchors)
      
      # switch to integrated assay. The variable features of this assay are automatically set during
      # IntegrateData
      DefaultAssay(rawdata.integrated) <- "integrated"
      # Run the standard workflow for visualization and clustering
      rawdata.integrated <- ScaleData(rawdata.integrated, verbose = FALSE)
      rawdata.integrated <- RunPCA(rawdata.integrated, npcs = 30, verbose = FALSE)
      folder_path = paste("/Users/Healthy/Desktop/final/scripts and results/Seurat/pcs/Imbalanced Datasets Involving Batch Effects", paste("ratio", ratio, sep = ""), paste("dropout", dropout, sep = ""), sep="/")
      file_path = paste(paste(folder_path, "pcs_duplicate", sep="/"), duplicate, ".csv", sep="")
      if (!file.exists(folder_path)){
        dir.create(folder_path, recursive = TRUE)
      }
      write.csv(rawdata.integrated@reductions$pca@cell.embeddings, file_path)
          }
        }
      }