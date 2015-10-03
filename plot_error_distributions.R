
# plot density curves of error distributions

data = read.csv('output-notech.csv', header= TRUE)
data = subset(data, word != "")

linewidth = 2 # default is 1
density.adjust = 1.5 # default is 1

data.RC = data$RC
data.LC = data$LC
data.RCLC = data$RCLC
data.BinaryFootRL = data$BinRL
data.BinaryFootLR = data$BinLR

minX = min(c(data.RC, data.LC, data.RCLC, data.BinaryFootRL, data.BinaryFootLR))
maxX = max(c(data.RC, data.LC, data.RCLC, data.BinaryFootRL, data.BinaryFootLR))

minY = min(as.data.frame(table(data.RC))[2], as.data.frame(table(data.LC))[2], 
           as.data.frame(table(data.RCLC))[2],
           as.data.frame(table(data.BinaryFootRL))[2],
           as.data.frame(table(data.BinaryFootLR))[2])

maxY = max(as.data.frame(table(data.RC))[2], as.data.frame(table(data.LC))[2], 
           as.data.frame(table(data.RCLC))[2],
           as.data.frame(table(data.BinaryFootRL))[2],
           as.data.frame(table(data.BinaryFootLR))[2])

RC.density = density(data.RC, adjust=density.adjust)
LC.density = density(data.LC, adjust=density.adjust)
RCLC.density = density(data.RCLC, adjust=density.adjust)
binftRL.density = density(data.BinaryFootRL, adjust=density.adjust)
binftLR.density = density(data.BinaryFootLR, adjust=density.adjust)

minX.density = min(c(RC.density$x, LC.density$x, RCLC.density$x,
                     binftRL.density$x, binftLR.density$x))
maxX.density = max(c(RC.density$x, LC.density$x, RCLC.density$x,
                     binftRL.density$x, binftLR.density$x))
minY.density = min(c(RC.density$y, LC.density$y, RCLC.density$y,
                     binftRL.density$y, binftLR.density$y))
maxY.density = max(c(RC.density$y, LC.density$y, RCLC.density$y,
                     binftRL.density$y, binftLR.density$y))

# plot for paper
postscript('plots_errors/error-distribution-density-notech.eps')
plot(RC.density, lty=2, xlim=c(minX.density, maxX.density),
    ylim=c(minY.density, maxY.density), lwd=linewidth,
    main="", ylab="Density", xlab="Error")
lines(LC.density, lty=3, lwd=linewidth, ann=FALSE)
lines(RCLC.density, lty=1, lwd=linewidth, ann=FALSE)
lines(binftRL.density, lty=4, lwd=linewidth, ann=FALSE)
lines(binftLR.density, lty=5, lwd=linewidth, ann=FALSE)
legend(minX.density+1, maxY.density-0.05,
    c("RC","LC", "RC+LC", "BinRL", "BinLR"), lty=c(2,3,1,4,5), lwd=linewidth)
dev.off()

# colored plot (using density)
postscript('plots_errors/error-distribution-density-colored-notech.eps')
plot(RC.density, lty=1, xlim=c(minX.density, maxX.density),
    ylim=c(minY.density, maxY.density), lwd=linewidth,
    main="Error distributions", ylab="Density", xlab="Error")
lines(LC.density, lty=2, lwd=linewidth, ann=FALSE)
lines(RCLC.density, lty=1, lwd=3, col="#800000", ann=FALSE) # color is maroon
lines(binftRL.density, lty=1, lwd=linewidth, col="#0000FF", ann=FALSE) # blue
lines(binftLR.density, lty=2, lwd=linewidth, col="#0000FF", ann=FALSE) # blue
legend(minX.density+1, maxY.density-0.05, c("RC","LC","RC+LC","BinRL","BinLR"),
    lty=c(1,2,1,1,2), lwd=linewidth,
    text.col=c("#000000", "#000000", "#800000", "#0000FF", "#0000FF"),
    col=c("#000000", "#000000", "#800000", "#0000FF", "#0000FF"))
dev.off()


###################
# plot individual histograms

fillcolor = 'gray'
ylabname = 'Count'
maxY.overall = max(maxY, maxY.density*100)

postscript('plots_errors/error-distribution-histogram-RC-notech.eps')
hist(data.RC, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=RC.density$x, y=RC.density$y*100, lwd=linewidth)
dev.off()

postscript('plots_errors/error-distribution-histogram-LC-notech.eps')
hist(data.LC, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=LC.density$x, y=LC.density$y*100, lwd=linewidth)
dev.off()

postscript('plots_errors/error-distribution-histogram-RCLC-notech.eps')
hist(data.RCLC, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=RCLC.density$x, y=RCLC.density$y*100, lwd=linewidth)
dev.off()

postscript('plots_errors/error-distribution-histogram-BinFtRL-notech.eps')
hist(data.BinaryFootRL, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=binftRL.density$x, y=binftRL.density$y*100, lwd=linewidth)
dev.off()

postscript('plots_errors/error-distribution-histogram-BinFtLR-notech.eps')
hist(data.BinaryFootLR, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=binftLR.density$x, y=binftLR.density$y*100, lwd=linewidth)
dev.off()

