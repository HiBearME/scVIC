######## load data  ###### 
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SIMLR_imbalanced_dropout1.rda")
ARI_SIMLR_imbalanced_dropout1= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SIMLR_imbalanced_dropout1.rda")
NMI_SIMLR_imbalanced_dropout1 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SIMLR_imbalanced_dropout2.rda")
ARI_SIMLR_imbalanced_dropout2= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SIMLR_imbalanced_dropout2.rda")
NMI_SIMLR_imbalanced_dropout2 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SIMLR_imbalanced_dropout3.rda")
ARI_SIMLR_imbalanced_dropout3= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SIMLR_imbalanced_dropout3.rda")
NMI_SIMLR_imbalanced_dropout3 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SIMLR_imbalanced_dropout4.rda")
ARI_SIMLR_imbalanced_dropout4= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SIMLR_imbalanced_dropout4.rda")
NMI_SIMLR_imbalanced_dropout4 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SIMLR_imbalanced_dropout5.rda")
ARI_SIMLR_imbalanced_dropout5= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SIMLR_imbalanced_dropout5.rda")
NMI_SIMLR_imbalanced_dropout5 = nmi

ARI_SIMLR_imbalanced = rep(0, 250)
dim(ARI_SIMLR_imbalanced) = c(5, 5, 10)
ARI_SIMLR_imbalanced[1:5,1,1:10] = ARI_SIMLR_imbalanced_dropout1
ARI_SIMLR_imbalanced[1:5,2,1:10] = ARI_SIMLR_imbalanced_dropout2
ARI_SIMLR_imbalanced[1:5,3,1:10] = ARI_SIMLR_imbalanced_dropout3
ARI_SIMLR_imbalanced[1:5,4,1:10] = ARI_SIMLR_imbalanced_dropout4
ARI_SIMLR_imbalanced[1:5,5,1:10] = ARI_SIMLR_imbalanced_dropout5
NMI_SIMLR_imbalanced = rep(0, 250)
dim(NMI_SIMLR_imbalanced) = c(5, 5, 10)
NMI_SIMLR_imbalanced[1:5,1,1:10] = NMI_SIMLR_imbalanced_dropout1
NMI_SIMLR_imbalanced[1:5,2,1:10] = NMI_SIMLR_imbalanced_dropout2
NMI_SIMLR_imbalanced[1:5,3,1:10] = NMI_SIMLR_imbalanced_dropout3
NMI_SIMLR_imbalanced[1:5,4,1:10] = NMI_SIMLR_imbalanced_dropout4
NMI_SIMLR_imbalanced[1:5,5,1:10] = NMI_SIMLR_imbalanced_dropout5

load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_CIDR_imbalanced_dropout1.rda")
ARI_CIDR_imbalanced_dropout1= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_CIDR_imbalanced_dropout1.rda")
NMI_CIDR_imbalanced_dropout1 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_CIDR_imbalanced_dropout2.rda")
ARI_CIDR_imbalanced_dropout2= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_CIDR_imbalanced_dropout2.rda")
NMI_CIDR_imbalanced_dropout2 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_CIDR_imbalanced_dropout3.rda")
ARI_CIDR_imbalanced_dropout3= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_CIDR_imbalanced_dropout3.rda")
NMI_CIDR_imbalanced_dropout3 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_CIDR_imbalanced_dropout4.rda")
ARI_CIDR_imbalanced_dropout4= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_CIDR_imbalanced_dropout4.rda")
NMI_CIDR_imbalanced_dropout4 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_CIDR_imbalanced_dropout5.rda")
ARI_CIDR_imbalanced_dropout5= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_CIDR_imbalanced_dropout5.rda")
NMI_CIDR_imbalanced_dropout5 = nmi

ARI_CIDR_imbalanced = rep(0, 250)
dim(ARI_CIDR_imbalanced) = c(5, 5, 10)
ARI_CIDR_imbalanced[1:5,1,1:10] = ARI_CIDR_imbalanced_dropout1
ARI_CIDR_imbalanced[1:5,2,1:10] = ARI_CIDR_imbalanced_dropout2
ARI_CIDR_imbalanced[1:5,3,1:10] = ARI_CIDR_imbalanced_dropout3
ARI_CIDR_imbalanced[1:5,4,1:10] = ARI_CIDR_imbalanced_dropout4
ARI_CIDR_imbalanced[1:5,5,1:10] = ARI_CIDR_imbalanced_dropout5
NMI_CIDR_imbalanced = rep(0, 250)
dim(NMI_CIDR_imbalanced) = c(5, 5, 10)
NMI_CIDR_imbalanced[1:5,1,1:10] = NMI_CIDR_imbalanced_dropout1
NMI_CIDR_imbalanced[1:5,2,1:10] = NMI_CIDR_imbalanced_dropout2
NMI_CIDR_imbalanced[1:5,3,1:10] = NMI_CIDR_imbalanced_dropout3
NMI_CIDR_imbalanced[1:5,4,1:10] = NMI_CIDR_imbalanced_dropout4
NMI_CIDR_imbalanced[1:5,5,1:10] = NMI_CIDR_imbalanced_dropout5


load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SOUP_imbalanced_dropout1.rda")
ARI_SOUP_imbalanced_dropout1= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SOUP_imbalanced_dropout1.rda")
NMI_SOUP_imbalanced_dropout1 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SOUP_imbalanced_dropout2.rda")
ARI_SOUP_imbalanced_dropout2= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SOUP_imbalanced_dropout2.rda")
NMI_SOUP_imbalanced_dropout2 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SOUP_imbalanced_dropout3.rda")
ARI_SOUP_imbalanced_dropout3= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SOUP_imbalanced_dropout3.rda")
NMI_SOUP_imbalanced_dropout3 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SOUP_imbalanced_dropout4.rda")
ARI_SOUP_imbalanced_dropout4= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SOUP_imbalanced_dropout4.rda")
NMI_SOUP_imbalanced_dropout4 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_SOUP_imbalanced_dropout5.rda")
ARI_SOUP_imbalanced_dropout5= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_SOUP_imbalanced_dropout5.rda")
NMI_SOUP_imbalanced_dropout5 = nmi

ARI_SOUP_imbalanced = rep(0, 250)
dim(ARI_SOUP_imbalanced) = c(5, 5, 10)
ARI_SOUP_imbalanced[1:5,1,1:10] = ARI_SOUP_imbalanced_dropout1
ARI_SOUP_imbalanced[1:5,2,1:10] = ARI_SOUP_imbalanced_dropout2
ARI_SOUP_imbalanced[1:5,3,1:10] = ARI_SOUP_imbalanced_dropout3
ARI_SOUP_imbalanced[1:5,4,1:10] = ARI_SOUP_imbalanced_dropout4
ARI_SOUP_imbalanced[1:5,5,1:10] = ARI_SOUP_imbalanced_dropout5
NMI_SOUP_imbalanced = rep(0, 250)
dim(NMI_SOUP_imbalanced) = c(5, 5, 10)
NMI_SOUP_imbalanced[1:5,1,1:10] = NMI_SOUP_imbalanced_dropout1
NMI_SOUP_imbalanced[1:5,2,1:10] = NMI_SOUP_imbalanced_dropout2
NMI_SOUP_imbalanced[1:5,3,1:10] = NMI_SOUP_imbalanced_dropout3
NMI_SOUP_imbalanced[1:5,4,1:10] = NMI_SOUP_imbalanced_dropout4
NMI_SOUP_imbalanced[1:5,5,1:10] = NMI_SOUP_imbalanced_dropout5

load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_RaceID_imbalanced_dropout1.rda")
ARI_RaceID_imbalanced_dropout1= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_RaceID_imbalanced_dropout1.rda")
NMI_RaceID_imbalanced_dropout1 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_RaceID_imbalanced_dropout2.rda")
ARI_RaceID_imbalanced_dropout2= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_RaceID_imbalanced_dropout2.rda")
NMI_RaceID_imbalanced_dropout2 = nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_RaceID_imbalanced_dropout3.rda")
ARI_RaceID_imbalanced_dropout3= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_RaceID_imbalanced_dropout3.rda")
NMI_RaceID_imbalanced_dropout3= nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_RaceID_imbalanced_dropout4.rda")
ARI_RaceID_imbalanced_dropout4= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_RaceID_imbalanced_dropout4.rda")
NMI_RaceID_imbalanced_dropout4= nmi
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/ARI_RaceID_imbalanced_dropout5.rda")
ARI_RaceID_imbalanced_dropout5= ari
load("/Users/Healthy/Student/Now/scVIC/scripts and results/classical methods/saved/NMI_RaceID_imbalanced_dropout5.rda")
NMI_RaceID_imbalanced_dropout5= nmi

ARI_RaceID_imbalanced = rep(0, 250)
dim(ARI_RaceID_imbalanced) = c(5, 5, 10)
ARI_RaceID_imbalanced[1:5,1,1:10] = ARI_RaceID_imbalanced_dropout1
ARI_RaceID_imbalanced[1:5,2,1:10] = ARI_RaceID_imbalanced_dropout2
ARI_RaceID_imbalanced[1:5,3,1:10] = ARI_RaceID_imbalanced_dropout3
ARI_RaceID_imbalanced[1:5,4,1:10] = ARI_RaceID_imbalanced_dropout4
ARI_RaceID_imbalanced[1:5,5,1:10] = ARI_RaceID_imbalanced_dropout5
NMI_RaceID_imbalanced = rep(0, 250)
dim(NMI_RaceID_imbalanced) = c(5, 5, 10)
NMI_RaceID_imbalanced[1:5,1,1:10] = NMI_RaceID_imbalanced_dropout1
NMI_RaceID_imbalanced[1:5,2,1:10] = NMI_RaceID_imbalanced_dropout2
NMI_RaceID_imbalanced[1:5,3,1:10] = NMI_RaceID_imbalanced_dropout3
NMI_RaceID_imbalanced[1:5,4,1:10] = NMI_RaceID_imbalanced_dropout4
NMI_RaceID_imbalanced[1:5,5,1:10] = NMI_RaceID_imbalanced_dropout5

ARI_SOUP_imbalanced_median = apply(ARI_SOUP_imbalanced, c(1,2), median)
NMI_SOUP_imbalanced_median = apply(NMI_SOUP_imbalanced, c(1,2), median)
ARI_CIDR_imbalanced_median = apply(ARI_CIDR_imbalanced, c(1,2), median)
NMI_CIDR_imbalanced_median = apply(NMI_CIDR_imbalanced, c(1,2), median)
ARI_SIMLR_imbalanced_median = apply(ARI_SIMLR_imbalanced, c(1,2), median)
NMI_SIMLR_imbalanced_median = apply(NMI_SIMLR_imbalanced, c(1,2), median)
ARI_RaceID_imbalanced_median = apply(ARI_RaceID_imbalanced, c(1,2), median)
NMI_RaceID_imbalanced_median = apply(NMI_RaceID_imbalanced, c(1,2), median)

dimnames(ARI_SOUP_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
ARI_SOUP_imbalanced_median = as.data.frame(as.table(ARI_SOUP_imbalanced_median), responseName="ARI")
ARI_SOUP_imbalanced_median["methods"] = rep("SOUP", 25)
dimnames(NMI_SOUP_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
NMI_SOUP_imbalanced_median = as.data.frame(as.table(NMI_SOUP_imbalanced_median), responseName="NMI")
NMI_SOUP_imbalanced_median["methods"] = rep("SOUP", 25)
dimnames(ARI_CIDR_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
ARI_CIDR_imbalanced_median = as.data.frame(as.table(ARI_CIDR_imbalanced_median), responseName="ARI")
ARI_CIDR_imbalanced_median["methods"] = rep("CIDR", 25)
dimnames(NMI_CIDR_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
NMI_CIDR_imbalanced_median = as.data.frame(as.table(NMI_CIDR_imbalanced_median), responseName="NMI")
NMI_CIDR_imbalanced_median["methods"] = rep("CIDR", 25)
dimnames(ARI_SIMLR_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
ARI_SIMLR_imbalanced_median = as.data.frame(as.table(ARI_SIMLR_imbalanced_median), responseName="ARI")
ARI_SIMLR_imbalanced_median["methods"] = rep("SIMLR", 25)
dimnames(NMI_SIMLR_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
NMI_SIMLR_imbalanced_median = as.data.frame(as.table(NMI_SIMLR_imbalanced_median), responseName="NMI")
NMI_SIMLR_imbalanced_median["methods"] = rep("SIMLR", 25)
dimnames(ARI_RaceID_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
ARI_RaceID_imbalanced_median = as.data.frame(as.table(ARI_RaceID_imbalanced_median), responseName="ARI")
ARI_RaceID_imbalanced_median["methods"] = rep("RaceID", 25)
dimnames(NMI_RaceID_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
NMI_RaceID_imbalanced_median = as.data.frame(as.table(NMI_RaceID_imbalanced_median), responseName="NMI")
NMI_RaceID_imbalanced_median["methods"] = rep("RaceID", 25)

ARI_scVI_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/ARI_scVI_Louvain_imbalanced_median.csv", header = FALSE))
NMI_scVI_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVI/saved/NMI_scVI_Louvain_imbalanced_median.csv", header = FALSE))
ARI_scVIC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/ARI_scVIC_imbalanced_median.csv", header = FALSE))
NMI_scVIC_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/NMI_scVIC_imbalanced_median.csv", header = FALSE))
ARI_scVIC_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/ARI_scVIC_Louvain_imbalanced_median.csv", header = FALSE))
NMI_scVIC_Louvain_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVIC/saved/NMI_scVIC_Louvain_imbalanced_median.csv", header = FALSE))
ARI_scVAE_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVAE/saved/ARI_scVAE_imbalanced_median.csv", header = FALSE))
NMI_scVAE_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scVAE/saved/NMI_scVAE_imbalanced_median.csv", header = FALSE))
ARI_scziDesk_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scziDesk/saved/ARI_scziDesk_imbalanced_median.csv", header = FALSE))
NMI_scziDesk_imbalanced_median = as.matrix(read.csv("/Users/Healthy/Student/Now/scVIC/scripts and results/scziDesk/saved/NMI_scziDesk_imbalanced_median.csv", header = FALSE))

dimnames(ARI_scVI_Louvain_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVI_Louvain_imbalanced_median = as.data.frame(as.table(ARI_scVI_Louvain_imbalanced_median), responseName="ARI")
ARI_scVI_Louvain_imbalanced_median["methods"] = rep("scVI-Louvain", 25)
dimnames(NMI_scVI_Louvain_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVI_Louvain_imbalanced_median = as.data.frame(as.table(NMI_scVI_Louvain_imbalanced_median), responseName="NMI")
NMI_scVI_Louvain_imbalanced_median["methods"] = rep("scVI-Louvain", 25)
dimnames(ARI_scVIC_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
ARI_scVIC_imbalanced_median = as.data.frame(as.table(ARI_scVIC_imbalanced_median), responseName="ARI")
ARI_scVIC_imbalanced_median["methods"] = rep("scVIC", 25)
dimnames(NMI_scVIC_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
NMI_scVIC_imbalanced_median = as.data.frame(as.table(NMI_scVIC_imbalanced_median), responseName="NMI")
NMI_scVIC_imbalanced_median["methods"] = rep("scVIC", 25)
dimnames(ARI_scVIC_Louvain_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
ARI_scVIC_Louvain_imbalanced_median = as.data.frame(as.table(ARI_scVIC_Louvain_imbalanced_median), responseName="ARI")
ARI_scVIC_Louvain_imbalanced_median["methods"] = rep("scVIC-Louvain", 25)
dimnames(NMI_scVIC_Louvain_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0.0, 0.5))
NMI_scVIC_Louvain_imbalanced_median = as.data.frame(as.table(NMI_scVIC_Louvain_imbalanced_median), responseName="NMI")
NMI_scVIC_Louvain_imbalanced_median["methods"] = rep("scVIC-Louvain", 25)
dimnames(ARI_scVAE_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
ARI_scVAE_imbalanced_median = as.data.frame(as.table(ARI_scVAE_imbalanced_median), responseName="ARI")
ARI_scVAE_imbalanced_median["methods"] = rep("scVAE", 25)
dimnames(NMI_scVAE_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
NMI_scVAE_imbalanced_median = as.data.frame(as.table(NMI_scVAE_imbalanced_median), responseName="NMI")
NMI_scVAE_imbalanced_median["methods"] = rep("scVAE", 25)
dimnames(ARI_scziDesk_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
ARI_scziDesk_imbalanced_median = as.data.frame(as.table(ARI_scziDesk_imbalanced_median), responseName="ARI")
ARI_scziDesk_imbalanced_median["methods"] = rep("scziDesk", 25)
dimnames(NMI_scziDesk_imbalanced_median) = list(ratio=c(0.6, 0.7, 0.8, 0.9, 1.0), dropout=c(-1.5, -1.0, -0.5, 0, 0.5))
NMI_scziDesk_imbalanced_median = as.data.frame(as.table(NMI_scziDesk_imbalanced_median), responseName="NMI")
NMI_scziDesk_imbalanced_median["methods"] = rep("scziDesk", 25)

ARI_imbalanced_median = rbind(ARI_SOUP_imbalanced_median, ARI_CIDR_imbalanced_median, ARI_SIMLR_imbalanced_median, ARI_RaceID_imbalanced_median, ARI_scziDesk_imbalanced_median, ARI_scVAE_imbalanced_median, ARI_scVI_Louvain_imbalanced_median, ARI_scVIC_imbalanced_median, ARI_scVIC_Louvain_imbalanced_median)
NMI_imbalanced_median = rbind(NMI_SOUP_imbalanced_median, NMI_CIDR_imbalanced_median, NMI_SIMLR_imbalanced_median, NMI_RaceID_imbalanced_median, NMI_scziDesk_imbalanced_median, NMI_scVAE_imbalanced_median, NMI_scVI_Louvain_imbalanced_median, NMI_scVIC_imbalanced_median, NMI_scVIC_Louvain_imbalanced_median)

ARI_imbalanced_median$ratio = as.character(ARI_imbalanced_median$ratio)
ARI_imbalanced_median$ratio = as.numeric(ARI_imbalanced_median$ratio)

NMI_imbalanced_median$ratio = as.character(NMI_imbalanced_median$ratio)
NMI_imbalanced_median$ratio = as.numeric(NMI_imbalanced_median$ratio)

ARI_imbalanced_median$methods = factor(ARI_imbalanced_median$methods, levels = c("scVAE", "RaceID", "SIMLR", "CIDR", "SOUP", "scVI-Louvain","scziDesk","scVIC","scVIC-Louvain"))
NMI_imbalanced_median$methods = factor(NMI_imbalanced_median$methods, levels = c("scVAE", "RaceID", "SIMLR", "CIDR", "SOUP", "scVI-Louvain","scziDesk","scVIC","scVIC-Louvain"))

library(ggplot2)
theme_set(theme_bw())
######## box plot######  
pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median.pdf", width=7, height=4)
ggplot(data = ARI_imbalanced_median, aes(x = methods, y = ARI, fill=methods)) +  geom_boxplot() + ggtitle("Imbalanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median.pdf", width=7, height=4)
ggplot(data = NMI_imbalanced_median, aes(x = methods, y = NMI, fill=methods)) +  geom_boxplot() + ggtitle("Imbalanced") + theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
dev.off()

dropout.degree <- c('5.6%', '8.2%', '11.9%', '16.8%', '23.0%')
######## line plot by ratio on imbalanced dataset###### 

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_ratio3.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.6,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.6 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_ratio3.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.6,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.6 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_ratio4.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.7,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.7 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_ratio4.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.7,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.7 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree)
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_ratio5.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.8,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.8 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_ratio5.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.8,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.8 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_ratio6.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 0.9,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.9 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_ratio6.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 0.9,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("0.9 as imbalanced ratio") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_ratio7.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["ratio"] == 1.0,], aes(x = dropout, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("1.0 as imbalanced ratio (actually balanced)") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_ratio7.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["ratio"] == 1.0,], aes(x = dropout, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("1.0 as imbalanced ratio (actually balanced)") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("dropout degree") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_discrete(labels = dropout.degree) 
dev.off()
######## line plot by dropout on imbalanced dataset###### 

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == -1.5,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_dropout1.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == -1.5,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 5.6%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == -1.0,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_dropout2.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == -1.0,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 8.2%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == -0.5,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_dropout3.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == -0.5,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 11.9%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == 0,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_dropout4.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == 0,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 16.8%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/ARI_imbalanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = ARI_imbalanced_median[ARI_imbalanced_median["dropout"] == 0.5,], aes(x = -ratio, y = ARI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()

pdf("/Users/Healthy/Student/Now/scVIC/figs and charts/imbalanced/NMI_imbalanced_median_dropout5.pdf", width=7, height=2.5)
ggplot(data = NMI_imbalanced_median[NMI_imbalanced_median["dropout"] == 0.5,], aes(x = -ratio, y = NMI, group=methods, colour=methods)) +  geom_line() + ggtitle("Dropout degree 23.0%") + theme(plot.title = element_text(hjust = 0.5), panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) + geom_point(size=4, shape=20) + geom_line(size=1.5) + xlab("ratio for imbalance") + scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) + scale_x_continuous(breaks = c(-1.0, -0.9, -0.8, -0.7, -0.6), labels = c(1.0, 0.9, 0.8, 0.7, 0.6))
dev.off()



