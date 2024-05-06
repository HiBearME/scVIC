ari_data_frame <- data.frame(Method = rep(c("scVIC_Louvain", "scVI_Louvain", "scziDesk", "CIDR", "RaceID", "SIMLR", "SOUP"), each=3),
                  Dataset = rep(c("Quake_10x_Trachea", "Tosches_turtle", "Bach"),time=7),
                  ARI = c(0.94379, 0.83464, 0.88382, 0.93214, 0.82207, 0.77912, 0.89084, 0.55807, 0.87302, 0.34009, 0.25858, 0.81942, 0.13811, 0.34514, 0.61745, 0.05246, 0.53846, 0.77467, 0.26950 , 0.10349, 0.45668))
ari_data_frame$Dataset = factor(ari_data_frame$Dataset, levels = c("Quake_10x_Trachea", "Tosches_turtle", "Bach"))
ari_data_frame$Method = factor(ari_data_frame$Method, levels = c("SOUP", "RaceID", "SIMLR", "CIDR", "scziDesk", "scVI_Louvain", "scVIC_Louvain"))
nmi_data_frame <- data.frame(Method = rep(c("scVIC_Louvain", "scVI_Louvain", "scziDesk", "CIDR", "RaceID", "SIMLR", "SOUP"), each=3),
                             Dataset = rep(c("Quake_10x_Trachea", "Tosches_turtle", "Bach"),time=7),
                             NMI = c(0.87011, 0.86106, 0.83205, 0.85656, 0.84559, 0.77557, 0.84577, 0.69647, 0.85056, 0.49663, 0.48253, 0.78721, 0.39063, 0.57116, 0.69790, 0.17801, 0.62735, 0.77271, 0.48341,  0.51842, 0.63582))
nmi_data_frame$Dataset = factor(nmi_data_frame$Dataset, levels = c("Quake_10x_Trachea", "Tosches_turtle", "Bach"))
nmi_data_frame$Method = factor(nmi_data_frame$Method, levels = c("SOUP", "RaceID", "SIMLR", "CIDR", "scziDesk", "scVI_Louvain", "scVIC_Louvain"))
library(ggplot2)
theme_set(theme_bw())
######## bar plot ###### 
pdf("/Users/Healthy/Desktop/Now/scVIC/figs and charts/biological datasets/ARI_barplot.pdf", width=7, height=4)
ggplot(data = ari_data_frame, aes(x = Dataset, y = ARI, fill=Method)) + 
  geom_bar(stat="identity", position=position_dodge(0.8), width=0.7) + 
  scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  theme(legend.position="bottom", legend.title = , panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  guides(fill = guide_legend(nrow = 1)) + labs(x=NULL, fill=NULL)
dev.off()

pdf("/Users/Healthy/Desktop/Now/scVIC/figs and charts/biological datasets/NMI_barplot.pdf", width=7, height=4)
ggplot(data = nmi_data_frame, aes(x = Dataset, y = NMI, fill=Method)) + 
  geom_bar(stat="identity", position=position_dodge(0.8), width=0.7) + 
  scale_y_continuous(limits=c(0, 1), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  theme(legend.position="bottom", legend.title = , panel.border = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  guides(fill = guide_legend(nrow = 1)) + labs(x=NULL, fill=NULL)
dev.off()
