ARI_scVIC_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/ARI_scVIC_balanced_involving_batch_effects_median.csv", header = FALSE))
NMI_scVIC_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/NMI_scVIC_balanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_scVIC_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/BatchMixing_scVIC_balanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_scVIC_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVIC_balanced_median = as.data.frame(as.table(ARI_scVIC_balanced_median), responseName="ARI")
ARI_scVIC_balanced_median["methods"] = rep("scVIC", 25)
dimnames(NMI_scVIC_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVIC_balanced_median = as.data.frame(as.table(NMI_scVIC_balanced_median), responseName="NMI")
NMI_scVIC_balanced_median["methods"] = rep("scVIC", 25)
dimnames(BatchMixing_scVIC_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_scVIC_balanced_median = as.data.frame(as.table(BatchMixing_scVIC_balanced_median), responseName="BatchMixing")
BatchMixing_scVIC_balanced_median["methods"] = rep("scVIC", 25)

ARI_scVIC_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/ARI_scVIC_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))
NMI_scVIC_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/NMI_scVIC_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_scVIC_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/BatchMixing_scVIC_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_scVIC_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVIC_Louvain_balanced_median = as.data.frame(as.table(ARI_scVIC_Louvain_balanced_median), responseName="ARI")
ARI_scVIC_Louvain_balanced_median["methods"] = rep("scVIC-Louvain", 25)
dimnames(NMI_scVIC_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVIC_Louvain_balanced_median = as.data.frame(as.table(NMI_scVIC_Louvain_balanced_median), responseName="NMI")
NMI_scVIC_Louvain_balanced_median["methods"] = rep("scVIC-Louvain", 25)
dimnames(BatchMixing_scVIC_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_scVIC_Louvain_balanced_median = as.data.frame(as.table(BatchMixing_scVIC_Louvain_balanced_median), responseName="BatchMixing")
BatchMixing_scVIC_Louvain_balanced_median["methods"] = rep("scVIC-Louvain", 25)

ARI_scVI_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/ARI_scVI_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))
NMI_scVI_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/NMI_scVI_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_scVI_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/BatchMixing_scVI_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_scVI_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVI_Louvain_balanced_median = as.data.frame(as.table(ARI_scVI_Louvain_balanced_median), responseName="ARI")
ARI_scVI_Louvain_balanced_median["methods"] = rep("scVI-Louvain", 25)
dimnames(NMI_scVI_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVI_Louvain_balanced_median = as.data.frame(as.table(NMI_scVI_Louvain_balanced_median), responseName="NMI")
NMI_scVI_Louvain_balanced_median["methods"] = rep("scVI-Louvain", 25)
dimnames(BatchMixing_scVI_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_scVI_Louvain_balanced_median = as.data.frame(as.table(BatchMixing_scVI_Louvain_balanced_median), responseName="BatchMixing")
BatchMixing_scVI_Louvain_balanced_median["methods"] = rep("scVI-Louvain", 25)

ARI_DESC_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/DESC/saved_new/ARI_DESC_balanced_involving_batch_effects_median.csv", header = FALSE))
NMI_DESC_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/DESC/saved_new/NMI_DESC_balanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_DESC_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/DESC/saved_new/BatchMixing_DESC_balanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_DESC_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_DESC_balanced_median = as.data.frame(as.table(ARI_DESC_balanced_median), responseName="ARI")
ARI_DESC_balanced_median["methods"] = rep("DESC", 25)
dimnames(NMI_DESC_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_DESC_balanced_median = as.data.frame(as.table(NMI_DESC_balanced_median), responseName="NMI")
NMI_DESC_balanced_median["methods"] = rep("DESC", 25)
dimnames(BatchMixing_DESC_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_DESC_balanced_median = as.data.frame(as.table(BatchMixing_DESC_balanced_median), responseName="BatchMixing")
BatchMixing_DESC_balanced_median["methods"] = rep("DESC", 25)

ARI_Seurat_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/Seurat/saved/ARI_Seurat_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))
NMI_Seurat_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/Seurat/saved/NMI_Seurat_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_Seurat_Louvain_balanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/Seurat/saved/BatchMixing_Seurat_Louvain_balanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_Seurat_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_Seurat_Louvain_balanced_median = as.data.frame(as.table(ARI_Seurat_Louvain_balanced_median), responseName="ARI")
ARI_Seurat_Louvain_balanced_median["methods"] = rep("Seurat-Louvain", 25)
dimnames(NMI_Seurat_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_Seurat_Louvain_balanced_median = as.data.frame(as.table(NMI_Seurat_Louvain_balanced_median), responseName="NMI")
NMI_Seurat_Louvain_balanced_median["methods"] = rep("Seurat-Louvain", 25)
dimnames(BatchMixing_Seurat_Louvain_balanced_median) = list(cluster=c(3, 4 ,5 ,6 ,7), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_Seurat_Louvain_balanced_median = as.data.frame(as.table(BatchMixing_Seurat_Louvain_balanced_median), responseName="BatchMixing")
BatchMixing_Seurat_Louvain_balanced_median["methods"] = rep("Seurat-Louvain", 25)

ARI_balanced_median = rbind(ARI_scVIC_balanced_median, ARI_scVIC_Louvain_balanced_median, ARI_scVI_Louvain_balanced_median, ARI_DESC_balanced_median, ARI_Seurat_Louvain_balanced_median)
NMI_balanced_median = rbind(NMI_scVIC_balanced_median, NMI_scVIC_Louvain_balanced_median, NMI_scVI_Louvain_balanced_median, NMI_DESC_balanced_median, NMI_Seurat_Louvain_balanced_median)
BatchMixing_balanced_median = rbind(BatchMixing_scVIC_balanced_median, BatchMixing_scVIC_Louvain_balanced_median, BatchMixing_scVI_Louvain_balanced_median, BatchMixing_DESC_balanced_median, BatchMixing_Seurat_Louvain_balanced_median)

ARI_balanced_median$methods = factor(ARI_balanced_median$methods, levels = c("DESC", "scVI-Louvain", "Seurat-Louvain", "scVIC","scVIC-Louvain"))
NMI_balanced_median$methods = factor(NMI_balanced_median$methods, levels = c("DESC", "scVI-Louvain", "Seurat-Louvain", "scVIC","scVIC-Louvain"))
BatchMixing_balanced_median$methods = factor(BatchMixing_balanced_median$methods, levels = c("DESC", "scVI-Louvain", "Seurat-Louvain", "scVIC","scVIC-Louvain"))


library(ggplot2)
theme_set(theme_bw())
######## box plot######  
pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median.pdf", width=7, height=4)
ggplot(data = ARI_balanced_median, aes(x = methods, y = ARI, fill=methods)) +  geom_boxplot() + ggtitle("Balanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median.pdf", width=7, height=4)
ggplot(data = NMI_balanced_median, aes(x = methods, y = NMI, fill=methods)) +  geom_boxplot() + ggtitle("Balanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
dev.off()


pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median.pdf", width=7, height=4)
ggplot(data = BatchMixing_balanced_median, aes(x = methods, y = BatchMixing, fill=methods)) +  geom_boxplot() + ggtitle("Balanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence")
dev.off()

dropout.degree <- c('5.6%', '8.2%', '11.9%', '16.8%', '23.0%')
######## line plot by cluster on balanced dataset###### 

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_cluster3.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["cluster"] == 3,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 3") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_cluster3.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["cluster"] == 3,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 3") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_cluster3.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["cluster"] == 3,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 3") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence") + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_cluster4.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["cluster"] == 4,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 4") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_cluster4.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["cluster"] == 4,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 4") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_cluster4.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["cluster"] == 4,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 4") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence") + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_cluster5.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["cluster"] == 5,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 5") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_cluster5.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["cluster"] == 5,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 5") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_cluster5.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["cluster"] == 5,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 5") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence") + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_cluster6.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["cluster"] == 6,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 6") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)   
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_cluster6.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["cluster"] == 6,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 6") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_cluster6.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["cluster"] == 6,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 6") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence") + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_cluster7.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["cluster"] == 7,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 7") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_cluster7.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["cluster"] == 7,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 7") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)  
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_cluster7.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["cluster"] == 7,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Cluster number 7") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence") + scale_x_discrete(labels = dropout.degree)  
dev.off()
######## line plot by dropout on balanced dataset###### 

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["dropout"] == -1.5,], aes(x = cluster, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["dropout"] == -1.5,], aes(x = cluster, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["dropout"] == -1.5,], aes(x = cluster, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["dropout"] == -1.0,], aes(x = cluster, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["dropout"] == -1.0,], aes(x = cluster, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["dropout"] == -1.0,], aes(x = cluster, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["dropout"] == -0.5,], aes(x = cluster, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["dropout"] == -0.5,], aes(x = cluster, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["dropout"] == -0.5,], aes(x = cluster, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["dropout"] == 0.0,], aes(x = cluster, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["dropout"] == 0.0,], aes(x = cluster, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["dropout"] == 0.0,], aes(x = cluster, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/ARI_balanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = ARI_balanced_median[ARI_balanced_median["dropout"] == 0.5,], aes(x = cluster, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/NMI_balanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = NMI_balanced_median[NMI_balanced_median["dropout"] == 0.5,], aes(x = cluster, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/balanced involving batch effects for new DESC/BatchMixing_balanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_balanced_median[BatchMixing_balanced_median["dropout"] == 0.5,], aes(x = cluster, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("cluster number") + scale_y_continuous(limits=c(0, 0.4), breaks=c(0.0, 0.1, 0.2, 0.3, 0.4)) + labs(y="KL divergence")
dev.off()