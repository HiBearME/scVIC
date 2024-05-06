library(Seurat)

path = "/Users/Healthy/Desktop/final/scripts and results/generate simulated datasets/data/Balanced Datasets Involving Batch Effects"
save_path = "/Users/Healthy/Desktop/final/scripts and results/Seurat/pcs/Balanced Datasets Involving Batch Effects"
dropouts <- c(3)

clusters <- seq(1, 5)
duplicates <- seq(1, 10)
for(cluster in clusters){
  for(dropout in dropouts){
    for(duplicate in duplicates){
      set.seed(1)
      path_ratios <- paste(path, cluster  + 2, sep = "/")
      path_dropout <- paste(path_ratios, paste("dropout", dropout, sep = ""), sep="/")
      counts_file <- paste(paste(path_dropout, "counts_duplicate", sep="/"), duplicate, ".csv", sep="")
      batches_file <- paste(paste(path_dropout, "batches_duplicate", sep="/"), duplicate, ".csv", sep="")
      rawdata <- read.csv(counts_file)
      rawdata <- rawdata[,2:2501] 
      batch <- read.csv(batches_file)
      batch <- batch[,2]

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

      DefaultAssay(rawdata.integrated) <- "integrated"
      rawdata.integrated <- ScaleData(rawdata.integrated, verbose = FALSE)
      rawdata.integrated <- RunPCA(rawdata.integrated, npcs = 30, verbose = FALSE)
      pca <- rawdata.integrated@reductions$pca@cell.embeddings
      save_path_ratios <- paste(save_path, cluster  + 2, sep = "/")
      save_path_dropout <- paste(save_path_ratios, paste("dropout", dropout, sep = ""), sep="/")
      save_path_pca<- paste(paste(save_path_dropout, "pcs_duplicate", sep="/"), duplicate, ".csv", sep="")

      ifelse(!dir.exists(save_path_ratios), dir.create(save_path_ratios), FALSE)  
      ifelse(!dir.exists(save_path_dropout), dir.create(save_path_dropout), FALSE)    
      write.csv2(pca, file=paste(save_path_pca))
    }
  }
}


