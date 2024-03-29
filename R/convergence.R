require(data.table)
require(ggplot2)
require(dplyr)
require(tidyr)
require(cowplot)
require(gtable)
require(facetscales)
require(ggpubr)
require(grid)
require(gridExtra)


combine_data<-function(x){
  # Combine vector and matrix data sets.
  data1.0<-fread(file.path(x, 'matrix','summary.angles.u.tsv'))
  colnames(data1.0)<- c('iteration', paste0('Eigenvector ', 1:10), 'sites', 'Algorithm', 'filename')
  data1.0<- data1.0 %>% pivot_longer(-c(iteration, sites, Algorithm, filename), names_to='eigenvector', values_to = 'angle')
  
  data1<- data1.0 %>%
    group_by(iteration, sites, Algorithm, eigenvector) %>% summarise(mean_angle= mean(angle))
  
  data1<-as.data.table(data1)
  data1$eigenvector<- sapply(data1$eigenvector, function(x) as.numeric(gsub('Eigenvector', '', x)))
  
  data2.0<-fread(file.path(x, 'vector','summary.angles.u.tsv'))
  colnames(data2.0)<-  c('eigenvector', 'iteration', 'angle' , 'sites', 'Algorithm', 'filename')
  data2.0[, index:=1:.N, by=c('sites', 'Algorithm', 'eigenvector', 'filename')]
  data2 <- data2.0 %>% group_by(index, sites, Algorithm, eigenvector) %>% summarise(mean_angle= mean(angle))
  data2<- as.data.table(data2)
  colnames(data2)<-colnames(data1)
  
  
  
  # Create an offset vector for vizualisation
  offset<-data2 %>% group_by(eigenvector, sites, Algorithm)%>% summarise(offset=max(iteration)) %>% summarise(mean_offset = floor(mean(offset)))
  offset$eigenvector<-offset$eigenvector+1
  offset$mean_offset<-offset$mean_offset-1
  
  #make sure the convergence line starts at the mean iteration to have one line
  data2<- data2 %>% left_join(offset)
  data2$iteration<-rowSums(data2[, c("iteration", "mean_offset")], na.rm = T)
  data2<-data2[, c("iteration", "sites", "Algorithm", "eigenvector", "mean_angle")]
  data2<-as.data.table(data2)
  
  # save a dataframe containing the iterations.
  iterations <-  rbind(data1.0, data2.0[, 1:6, with=F]) %>% group_by(sites, Algorithm, filename) %>% summarise(maxit = max(iteration))
  
  
  # fill the data frame with 90 before the start of gradient convergence
  expanded<-list()
  for (i in 1:(length(offset$eigenvector)-1)){
    print(i)
    expanded[[i]]<-data.table(seq(1,  offset$mean_offset[i]), rep(5, offset$mean_offset[i]), rep('gradient', offset$mean_offset[i]), rep(offset$eigenvector[i], offset$mean_offset[i]), rep(90, offset$mean_offset[i]))
  }
  
  # Combine the data sets
  data<- rbind(data1, data2)
  
  expanded<-rbindlist(expanded)
  data<- rbind(data, expanded, use.names=F)
  
  # add nice facet labels
  data$eigenvector<-paste0('Eigenvector ', data$eigenvector)
  data$Algorithm<-recode(data$Algorithm, 'power-approx'='AI-FULL', 'power-iteration'='RI-FULL', 'randomized'='RAND', 'gradient'='GUO')
  
  return(list(iterations, data))
}

ob<-combine_data('/home/anne/Documents/featurecloud/singular-value-decomposition/results/mnist/')

# manually fix the scales for the convergence plots
scales_x <- list(
  `Eigenvector 1` = scale_x_continuous(limits = c(0, 30)),
  `Eigenvector 5` = scale_x_continuous(limits = c(0, 300)),
  `Eigenvector 10` =scale_x_continuous(limits = c(0,1100))
  
)

#manually set the breaks for the convergence plots
breaks_fun <- function(x) {
  if (max(x)>1000){
    seq(500, max(x), 500)
  }
  else if (max(x) > 500) {
    seq(250, max(x), 250)
  } else if (max(x)> 250) {
    seq(100, max(x), 150)
  }
  else{
    seq(50, max(x), 100)
  }
}
# make a nice minimal theme
extra_theme <-theme(axis.text.y = element_text( size=8),
                    axis.title = element_text(size=9),
                    plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),
                    legend.margin = margin(t = -12, r = 0, b = 0, l = 0),
                    axis.text.x = element_text(angle = 90 , size = 8, hjust = 1, vjust = 0.5, margin = margin(t = 1, r = 0, b = 0, l = 0)))

# plot the convergence lines.
conv<-ggplot(data[eigenvector %in% c('Eigenvector 1', 'Eigenvector 5', 'Eigenvector 10')], 
             aes(iteration, mean_angle, group=interaction(Algorithm, Algorithm), color=Algorithm))+
  geom_line(aes(linetype=Algorithm))+
  xlab('Iterations')+
  theme_classic()+
  ylab('Angles w.r.t reference')+
  facet_grid_sc(rows=vars(SNPs), cols = vars(eigenvector), scales = list(x = scales_x))+
  scale_x_continuous('Algorithm',breaks=breaks_fun, limits = c(0, NA))+
  theme(legend.position = 'bottom',
        strip.text = element_text(size=8), panel.border = element_rect(fill=NA), 
        panel.spacing = unit(0.05, 'cm'),strip.background = element_blank(), 
        strip.text.y = element_blank())+ # remove sites strip label
  scale_color_viridis_d(option = 'A', end=0.7)+extra_theme
conv



#Make side panel
# Convert the plot to a grob
text = 'MNIST'
size=10
gt <- ggplotGrob(conv)
# Get the positions of the right strips in the layout: t = top, l = left, ...
strip <-c(subset(gt$layout, grepl("strip-r", gt$layout$name), select = t:r))
# Text grob
text.grob = textGrob(text, rot = -90,
                     gp = gpar(fontsize = size))
# New olumn to the right of current strip
# Adjusts its width to text size
width = unit(2, "grobwidth", text.grob) + unit(1, "lines")
gt <- gtable_add_cols(gt, width, max(strip$r))  
# Add text grob to new column
gt <- gtable_add_grob(gt, text.grob, 
                      t = min(strip$t), l = max(strip$r) + 1, b = max(strip$b))
# Draw it
conv<-ggplotify::as.ggplot(gt)
conv

ggsave(conv,file='/home/anne/Documents/manuscripts/vertical-pattern-recognition/figures/convergence.pdf', height=6, units = 'cm', width = 15 )




extra_theme <-theme(axis.text.y = element_text( size=8),
                    axis.title = element_text(size=9),
                    plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),
                    legend.margin = margin(t = -12, r = 0, b = 0, l = 0),
                    axis.text.x = element_text(angle = 90 , size = 8, hjust = 1, vjust = 0.5, margin = margin(t = 1, r = 0, b = -12, l = 0)))



eog<- fread('~/Documents/featurecloud/singular-value-decomposition/results/mnist/matrix/summary.eo.tsv')
eo<-fread('/home/anne/Documents/featurecloud/singular-value-decomposition/results/mnist/vector/summary.eo.tsv')
eo<-rbind(eo, eog, use.names=F)
eo$V4<-recode(eo$V4, 'power-approx'='AI-FULL', 'power-iteration'='RI-FULL', 'randomized'='RAND', 'gradient'='GUO')
eo<- eo[V4 %in% c('AI-FULL', 'RI-FULL', 'RAND', 'GUO') ]
eo <- eo %>% group_by(V3,V4,V5) %>% summarise(sum=sum(V2))
pl.eo<-ggplot(eo, aes(V4, sum, fill=interaction(V4,V3), color=V4))+geom_boxplot()+ylab('Runtime[s] (Matrix oper.)')+xlab('')+
  theme_classic()+
  extra_theme+
  guides(fill=FALSE, color=F)+
  scale_color_viridis_d(option = 'A', end=0.7)+
  scale_fill_manual(values=c(viridisLite::magma(5, end=0.7, alpha = 0.6), viridisLite::magma(5, end=0.7, alpha = 0.8)))
  
  
pl.eo


iterations<-as.data.table(iterations)
iterations$Algorithm<-recode(iterations$Algorithm, 'power-approx'='AI-FULL', 'power-iteration'='RI-FULL', 'randomized'='RAND', 'gradient'='GUO')
iterations <- iterations[Algorithm %in% c('AI-FULL', 'RI-FULL', 'RAND', 'GUO') ]
pl.iter<-ggplot(iterations, aes(Algorithm, maxit, fill=interaction(Algorithm, sites),color=Algorithm))+
  geom_boxplot()+
  xlab('')+
  theme_classic()+
  ylab('Iterations to convergence')+
  extra_theme +
  guides(fill="none", color="none")+
  scale_color_viridis_d(option = 'A', end=0.7)+
  scale_fill_manual(values=c(viridisLite::magma(5, end=0.7, alpha = 0.6), viridisLite::magma(5, end=0.7, alpha = 0.8)))
pl.iter




transmission<-fread('~/Documents/featurecloud/singular-value-decomposition/results/mnist/matrix/summary.transmission.tsv')
transmission_guo<-fread('~/Documents/featurecloud/singular-value-decomposition/results/mnist/vector/summary.transmission.tsv')
transmission<-rbind(transmission_guo, transmission)
transmission<-transmission[V2 %in% c("H_local=CS", "H_global=SC")]
colnames(transmission) <- c('index', 'key', 'iteration', 'site', 'size', 'kA', 'sites', 'algorithm', 'filename')
transmission<-transmission[site==1]
volume<-transmission %>% select(size, sites, algorithm, filename)%>% group_by(sites, algorithm, filename)%>% summarise(total_volume=(sum(size)*4)/1000000000)

volume<-as.data.table(volume)
volume$algorithm<-recode(volume$algorithm, 'power-approx'='AI-FULL', 'power-iteration'='RI-FULL', 'randomized'='RAND', 'gradient'='GUO')
volume<- volume[algorithm %in% c( 'RI-FULL', 'AI-FULL', 'RAND', 'GUO')]
pl.volume<-ggplot(volume, aes(algorithm, total_volume, fill=interaction(algorithm, sites),color=algorithm))+geom_boxplot()+
  ylab('Total transmitted\n data [GB]')+
  xlab('')+
  theme_classic()+
  extra_theme+
  guides(fill=FALSE, color=F)+scale_color_viridis_d(option = 'A', end=0.7)+
  scale_fill_manual(values=c(viridisLite::magma(5, end=0.7, alpha = 0.6), viridisLite::magma(5, end=0.7, alpha = 0.8)))
pl.volume
############FAKE ALPHA LEGEND
pl.volume.legend<-ggplot(volume, aes(algorithm, total_volume, fill=algorithm, alpha=as.factor(sites)))+geom_boxplot()+
  theme_classic()+ylab('Total transmitted data [GB]')+xlab('')+
  scale_alpha_manual(values=c(0.2,0.7))+
  # Use this to create a legens for splits category
  #guides(fill=F,alpha=guide_legend('#Sites', override.aes=list(fill=hcl(c(15,195),100,0,alpha=c(0.2,0.7)))))
  guides(fill=F,alpha=guide_legend('#Sites', override.aes=list(fill=hcl(c(15),100,0,alpha=c(0.2)))))
  
pl.volume.legend


legend_b <- get_legend(
  pl.volume.legend +theme(legend.position = 'bottom',
                          legend.box.margin = margin(t = 0, r = 0, b = 0, l = 0),
                          legend.text = element_text(size = 8),
                          legend.title = element_text(size = 8))
)
text = 'MNIST'
size=10
text.grob = textGrob(text, rot = -90,
                     gp = gpar(fontsize = size))
gp<-ggarrange(pl.iter, pl.eo, pl.volume, ncol=3)
gp<-plot_grid(gp, text.grob, rel_widths = c(1,0.05))
gp<-plot_grid(gp, legend_b, ncol=1, rel_heights = c(1,0.1))
# Convert the plot to a grob
gp
ggsave(gp,file='/home/anne/Documents/manuscripts/vertical-pattern-recognition/figures/benchmark.pdf', height=6, units = 'cm', width = 15)

