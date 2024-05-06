ARI_scVIC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/ARI_scVIC_imbalanced_involving_batch_effects_median.csv", header = FALSE))
NMI_scVIC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/NMI_scVIC_imbalanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_scVIC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/BatchMixing_scVIC_imbalanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_scVIC_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVIC_imbalanced_median = as.data.frame(as.table(ARI_scVIC_imbalanced_median), responseName="ARI")
ARI_scVIC_imbalanced_median["methods"] = rep("scVIC", 25)
dimnames(NMI_scVIC_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVIC_imbalanced_median = as.data.frame(as.table(NMI_scVIC_imbalanced_median), responseName="NMI")
NMI_scVIC_imbalanced_median["methods"] = rep("scVIC", 25)
dimnames(BatchMixing_scVIC_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_scVIC_imbalanced_median = as.data.frame(as.table(BatchMixing_scVIC_imbalanced_median), responseName="BatchMixing")
BatchMixing_scVIC_imbalanced_median["methods"] = rep("scVIC", 25)

ARI_scVIC_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/ARI_scVIC_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))
NMI_scVIC_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/NMI_scVIC_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_scVIC_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/BatchMixing_scVIC_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_scVIC_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVIC_Louvain_imbalanced_median = as.data.frame(as.table(ARI_scVIC_Louvain_imbalanced_median), responseName="ARI")
ARI_scVIC_Louvain_imbalanced_median["methods"] = rep("scVIC-Louvain", 25)
dimnames(NMI_scVIC_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVIC_Louvain_imbalanced_median = as.data.frame(as.table(NMI_scVIC_Louvain_imbalanced_median), responseName="NMI")
NMI_scVIC_Louvain_imbalanced_median["methods"] = rep("scVIC-Louvain", 25)
dimnames(BatchMixing_scVIC_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_scVIC_Louvain_imbalanced_median = as.data.frame(as.table(BatchMixing_scVIC_Louvain_imbalanced_median), responseName="BatchMixing")
BatchMixing_scVIC_Louvain_imbalanced_median["methods"] = rep("scVIC-Louvain", 25)

ARI_scVI_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/ARI_scVI_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))
NMI_scVI_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/NMI_scVI_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_scVI_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/BatchMixing_scVI_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_scVI_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVI_Louvain_imbalanced_median = as.data.frame(as.table(ARI_scVI_Louvain_imbalanced_median), responseName="ARI")
ARI_scVI_Louvain_imbalanced_median["methods"] = rep("scVI-Louvain", 25)
dimnames(NMI_scVI_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVI_Louvain_imbalanced_median = as.data.frame(as.table(NMI_scVI_Louvain_imbalanced_median), responseName="NMI")
NMI_scVI_Louvain_imbalanced_median["methods"] = rep("scVI-Louvain", 25)
dimnames(BatchMixing_scVI_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_scVI_Louvain_imbalanced_median = as.data.frame(as.table(BatchMixing_scVI_Louvain_imbalanced_median), responseName="BatchMixing")
BatchMixing_scVI_Louvain_imbalanced_median["methods"] = rep("scVI-Louvain", 25)

ARI_DESC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/DESC/saved_new/ARI_DESC_imbalanced_involving_batch_effects_median.csv", header = FALSE))
NMI_DESC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/DESC/saved_new/NMI_DESC_imbalanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_DESC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/DESC/saved_new/BatchMixing_DESC_imbalanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_DESC_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_DESC_imbalanced_median = as.data.frame(as.table(ARI_DESC_imbalanced_median), responseName="ARI")
ARI_DESC_imbalanced_median["methods"] = rep("DESC", 25)
dimnames(NMI_DESC_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_DESC_imbalanced_median = as.data.frame(as.table(NMI_DESC_imbalanced_median), responseName="NMI")
NMI_DESC_imbalanced_median["methods"] = rep("DESC", 25)
dimnames(BatchMixing_DESC_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_DESC_imbalanced_median = as.data.frame(as.table(BatchMixing_DESC_imbalanced_median), responseName="BatchMixing")
BatchMixing_DESC_imbalanced_median["methods"] = rep("DESC", 25)

ARI_Seurat_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/Seurat/saved/ARI_Seurat_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))
NMI_Seurat_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/Seurat/saved/NMI_Seurat_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))
BatchMixing_Seurat_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/Seurat/saved/BatchMixing_Seurat_Louvain_imbalanced_involving_batch_effects_median.csv", header = FALSE))

dimnames(ARI_Seurat_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_Seurat_Louvain_imbalanced_median = as.data.frame(as.table(ARI_Seurat_Louvain_imbalanced_median), responseName="ARI")
ARI_Seurat_Louvain_imbalanced_median["methods"] = rep("Seurat-Louvain", 25)
dimnames(NMI_Seurat_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_Seurat_Louvain_imbalanced_median = as.data.frame(as.table(NMI_Seurat_Louvain_imbalanced_median), responseName="NMI")
NMI_Seurat_Louvain_imbalanced_median["methods"] = rep("Seurat-Louvain", 25)
dimnames(BatchMixing_Seurat_Louvain_imbalanced_median) = list(ratio = c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
BatchMixing_Seurat_Louvain_imbalanced_median = as.data.frame(as.table(BatchMixing_Seurat_Louvain_imbalanced_median), responseName="BatchMixing")
BatchMixing_Seurat_Louvain_imbalanced_median["methods"] = rep("Seurat-Louvain", 25)

ARI_imbalanced_median = rbind(ARI_scVIC_imbalanced_median, ARI_scVIC_Louvain_imbalanced_median, ARI_scVI_Louvain_imbalanced_median, ARI_DESC_imbalanced_median, ARI_Seurat_Louvain_imbalanced_median)
NMI_imbalanced_median = rbind(NMI_scVIC_imbalanced_median, NMI_scVIC_Louvain_imbalanced_median, NMI_scVI_Louvain_imbalanced_median, NMI_DESC_imbalanced_median, NMI_Seurat_Louvain_imbalanced_median)
BatchMixing_imbalanced_median = rbind(BatchMixing_scVIC_imbalanced_median, BatchMixing_scVIC_Louvain_imbalanced_median, BatchMixing_scVI_Louvain_imbalanced_median, BatchMixing_DESC_imbalanced_median, BatchMixing_Seurat_Louvain_imbalanced_median)

ARI_imbalanced_median$ratio = as.character(ARI_imbalanced_median$ratio)
ARI_imbalanced_median$ratio = as.numeric(ARI_imbalanced_median$ratio)

NMI_imbalanced_median$ratio = as.character(NMI_imbalanced_median$ratio)
NMI_imbalanced_median$ratio = as.numeric(NMI_imbalanced_median$ratio)

BatchMixing_imbalanced_median$ratio = as.character(BatchMixing_imbalanced_median$ratio)
BatchMixing_imbalanced_median$ratio = as.numeric(BatchMixing_imbalanced_median$ratio)

ARI_imbalanced_median$methods = factor(ARI_imbalanced_median$methods, levels = c("DESC", "scVI-Louvain", "Seurat-Louvain", "scVIC","scVIC-Louvain"))
NMI_imbalanced_median$methods = factor(NMI_imbalanced_median$methods, levels = c("DESC", "scVI-Louvain", "Seurat-Louvain", "scVIC","scVIC-Louvain"))
BatchMixing_imbalanced_median$methods = factor(BatchMixing_imbalanced_median$methods, levels = c("DESC", "scVI-Louvain", "Seurat-Louvain", "scVIC","scVIC-Louvain"))


library(ggplot2)
theme_set(theme_bw())
######## box plot######  
pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median.pdf", width=7, height=4)
ggplot(data = ARI_imbalanced_median, aes(x = methods, y = ARI, fill=methods)) +  geom_boxplot() + ggtitle("Imbalanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median.pdf", width=7, height=4)
ggplot(data = NMI_imbalanced_median, aes(x = methods, y = NMI, fill=methods)) +  geom_boxplot() + ggtitle("Imblanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median.pdf", width=7, height=4)
ggplot(data = BatchMixing_imbalanced_median, aes(x = methods, y = BatchMixing, fill=methods)) +  geom_boxplot() + ggtitle("Imblanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + labs(y="KL divergence")
dev.off()

dropout.degree <- c('5.6%', '8.2%', '11.9%', '16.8%', '23.0%')
######## line plot by ratio on imbalanced dataset###### 

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_ratio1.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.6,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.6 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_ratio1.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.6,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.6 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_ratio1.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["ratio"] == 0.6,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("0.6 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_ratio2.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.7,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.7 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_ratio2.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.7,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.7 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_ratio2.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["ratio"] == 0.7,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("0.7 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_ratio3.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.8,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.8 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_ratio3.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.8,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.8 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_ratio3.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["ratio"] == 0.8,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("0.8 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_ratio4.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.9,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.9 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_ratio4.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.9,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.9 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_ratio4.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["ratio"] == 0.9,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("0.9 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_ratio5.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 1.0,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("1.0 as imbalanced ratio (actually balanced)") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_ratio5.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 1.0,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("1.0 as imbalanced ratio (actually balanced)") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_ratio5.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["ratio"] == 1.0,], aes(x = dropout, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("1.0 as imbalanced ratio (actually balanced)") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))  + scale_x_discrete(labels = dropout.degree) + labs(y="KL divergence")
dev.off()
######## line plot by dropout on imbalanced dataset###### 

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == -1.5,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == -1.5,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["dropout"] == -1.5,], aes(x = -ratio, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == -1.0,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == -1.0,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["dropout"] == -1.0,], aes(x = -ratio, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == -0.5,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == -0.5,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["dropout"] == -0.5,], aes(x = -ratio, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == 0.0,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == 0.0,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["dropout"] == 0.0,], aes(x = -ratio, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") + labs(y="KL divergence")
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/ARI_imbalanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == 0.5,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/NMI_imbalanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == 0.5,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced involving batch effects for new DESC/BatchMixing_imbalanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = BatchMixing_imbalanced_median[BatchMixing_imbalanced_median["dropout"] == 0.5,], aes(x = -ratio, y = BatchMixing, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6)) + xlab("ratio for imbalance") + labs(y="KL divergence")
dev.off()


