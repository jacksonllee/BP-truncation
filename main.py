# Modeling Brazilian Portuguese
# Mike Pham, Jackson Lee, 2014-15

import math
import string
import subprocess
import codecs
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np

def elbowPoint(points):
	'''
	takes a list of points
	and outputs the index of the list for maximal curvature
	'''
	secondDerivativeList = [points[i+1] + points[i-1] - 2*points[i]
							for i in range(1, len(points) - 1)]
	secondDerivativeListPointTuples = sorted(enumerate(secondDerivativeList),
										key=lambda x:x[1], reverse=True)
	return secondDerivativeListPointTuples[0][0] + 2

def intersectionClosest(points1, points2):
#   points1[idx1] *   * points1[idx2]
#				  \ /
#				   *  <--- intersection point
#				  / \
#   points2[idx1] *   * points2[idx2]
	for (idx, (p1, p2)) in enumerate(zip(points1, points2)):
		if p2 > p1:
			idx1 = idx - 1
			idx2 = idx

			diff1 = abs(points1[idx1] - points2[idx1])
			diff2 = abs(points1[idx2] - points2[idx2])

			if diff1 <= diff2:
				return idx
			else:
				return idx+1


#--------------------------------------------------------------------#
##	Choose which orthographic representation system to use as input ##
##  and whether to include technical terms						  ##

asuffix = 'n'
techterms = 'n'
orthosys = 'k'

asuffix = raw_input("Treat -a in TF as part of TS (maximal symbol overlap)? Y/N ").lower()
if asuffix != "y":
	techterms = raw_input("Include technical terms? Y/N ").lower()
	if techterms != "n":
		orthosys = raw_input("Orthography system: k or c? ")

wordplotsDir = 'plots_words/'
errorplotsDir = 'plots_errors/'

#------------------------------------------------------------------------------#
###	determine file suffix		###

if asuffix == "y":
	filesuffix = '-a'
elif techterms == "n":
	filesuffix = '-notech'
elif orthosys == 'c':
	filesuffix = '-c'
else:
	filesuffix = ''

#------------------------------------------------------------------------------#
###	reads in lexicon, input full forms		###

# TODO: filesuffix is to work all input files (pt_br, goldstandard) plus output files (outlatex, output.csv

if orthosys == 'c':
	lexicon = codecs.open('pt_br_orthofix-c.txt', encoding='utf-8').read().split('\r\n')
else:
	lexicon = codecs.open('pt_br_orthofix.txt', encoding='utf-8').read().split('\r\n')

goldStandardFilename = 'goldStandard_binRL_orthofix%s.txt' % (filesuffix)
goldStandardOGbin = 'goldStandard_binLR_orthofix%s.txt' % (filesuffix)

dataList = [x[:x.index('\t')].replace('\n','').replace('#','')
			for x in codecs.open(goldStandardFilename, encoding='utf-8')]
dataListOGbin = [x[:x.index('\t')].replace('\n','').replace('#','')
			for x in codecs.open(goldStandardOGbin, encoding='utf-8')]

testwords = [x.replace('|','').replace('$','') for x in dataList]
goldStandard = [x.replace('$','') for x in dataList]
goldStandardBinFootRL = [x.replace('|','') for x in dataList]
goldStandardOGBinFoot = [x.replace('|','') for x in dataListOGbin]

lexDict = { x.split()[0]:int(x.split()[1]) for x in lexicon }
lexKeys = lexDict.keys()

#------------------------------------------------------------------------------#
##	set up output files		###

Rscriptname = wordplotsDir + 'generate_plots.R'
Rscript = codecs.open(Rscriptname, 'w', encoding='utf-8')

output = codecs.open('output%s.csv' % (filesuffix), 'w', encoding='utf-8')
outTex = codecs.open('outlatex%s.tex' % (filesuffix), 'w', encoding='utf-8')

#------------------------------------------------------------------------------#
###	initialize LaTeX .tex file	###

outTex.write('\\documentclass{article}\n')
outTex.write('\\usepackage{booktabs}\n')
outTex.write('\\usepackage{color}\n')
outTex.write('\\usepackage[letterpaper, landscape, margin=.5in]{geometry}\n')
outTex.write('\\usepackage{fontspec}\n')
outTex.write('\\setlength{\\parindent}{0em}\n')
outTex.write('\\begin{document}\n')

#------------------------------------------------------------------------------#
###	initialize variables	###

SFpredictList = list()
PFpredictList = list()
SFPFpredictClosestList = list()
truncPointList = list()
truncPointBinFootRLList = list()
truncPointOGBinFootList = list()

counter = 1

###############################################################################
###	This builds up the full form incrementally from the first letter,		###
###	checking each time to see how many words in the lexicon can be formed	###
### from that truncated form												###
###############################################################################

for (fullform, gold, goldBinFootRL, goldOGBinFoot) in zip(testwords, goldStandard, goldStandardBinFootRL, goldStandardOGBinFoot):
	print counter, fullform

	# get (i) gold standard trunc point and (ii) predicted trunc points by both bin foot models
	truncPoint = gold.index('|')
	truncPointList.append(truncPoint)

	truncPointBinFootRL = goldBinFootRL.index('$')
	truncPointBinFootRLList.append(truncPointBinFootRL)

	truncPointOGBinFoot = goldOGBinFoot.index('$')
	truncPointOGBinFootList.append(truncPointOGBinFoot)

	# initialize reversed form
	fullformReversed = fullform[::-1]
	trunc = ''
	truncReversed = ''

	# initialize LaTeX output strings
	truncdisplayTex = 'trunc:' + ' & '
	perletterdisplayTex = 'R-complete count:' + ' & '
	logcountdisplayTex = 'log(RC):' + ' & '
	perletterPFdisplayTex = ''
	logcountPFdisplayTex = ''

	# get the frequency of the full form in the corpus	##
	try:
		fullrank = lexDict[fullform]
	except KeyError:
		print '!!!!!', fullform
		fullrank = 0
#		continue

	logcountList = list()
	logcountReversedList = list()

	# loop through the lexicon to compute the counts of left- and right-completes
	for (e, (letter, letterReversed)) in enumerate(zip(fullform,
														fullformReversed)):
		count = 0
		countReversed = 0
		trunc = trunc + letter
		truncReversed = truncReversed + letterReversed

		if e < truncPoint:
			letterTex = '{\\color{red}\\bf ' + letter + '}'
		else:
			letterTex = letter
		truncdisplayTex = truncdisplayTex + letterTex + ' & '

		
		for word in lexKeys:
			wordReversed = word[::-1]

			######################################################################
			##	This counts the number R-completes given a TS			   	##
			######################################################################
			if word.startswith(trunc):
				count += 1
				
				################################################################
				##	This figures out what the frequency rank of the original  ##
				##	is relative to all completes from the trunc				  ##
				################################################################

#				wordrank = lexDict[word]
#				if wordrank > fullrank:
#					truncrank = truncrank + 1

			if wordReversed.startswith(truncReversed):
				countReversed += 1

		##################################################################
		##	This gets the log10 of the number of R- and L-completes		##
		##################################################################		
		if count > 0:
			logcount = round(math.log(count,10), 5)
		else:
			logcount = 0
		
		if countReversed > 0:
			logcountReversed = round(math.log(countReversed,10), 5)
		else:
			logcountReversed = 0

		logcountList.append(logcount)
		logcountReversedList.append(logcountReversed)

		######################################
		##	Builds nicer display of numbers	##
		######################################

		perletterdisplayTex = perletterdisplayTex + str(count) + ' & '
		logcountdisplayTex = logcountdisplayTex + str(logcount) + ' & '
		perletterPFdisplayTex = ' & ' + str(countReversed) + perletterPFdisplayTex
		logcountPFdisplayTex =  ' & ' + str(logcountReversed) + logcountPFdisplayTex

	# LaTeX output file
	outTex.write('\n\n%d\n\n' % (counter))
	outTex.write('{0}\n\n{1}\n\n\\vspace{{1em}}\n\n'.format(fullform,
													gold.replace('|','$|$')))
	outTex.write('\\begin{tabular}{l|%s}\n\n' % ('l'*len(fullform)))

	outTex.write(truncdisplayTex[:-3] + ' \\\\ \n')
	outTex.write(perletterdisplayTex[:-3] + ' \\\\ \n')
	outTex.write(logcountdisplayTex[:-3] + ' \\\\ \n')
	outTex.write('L-complete count:' + perletterPFdisplayTex + ' \\\\ \n')
	outTex.write('log(LC):' + logcountPFdisplayTex + ' \\\\ \n')

	outTex.write('\\end{tabular}\n\n')

	outTex.write('trunc point: ' + str(truncPoint) + '\n\n')



	SFpredict = elbowPoint(logcountList)
	PFpredict = elbowPoint(logcountReversedList[::-1])
	SFPFpredictClosest = intersectionClosest(logcountList, logcountReversedList[::-1])

	SFpredictList.append(SFpredict)
	PFpredictList.append(PFpredict)
	SFPFpredictClosestList.append(SFPFpredictClosest)

	outTex.write('RC point: ' + str(SFpredict)+'\n\n')
	outTex.write('LC point: ' + str(PFpredict)+'\n\n')

	outTex.write('RC-LC point: ' + str(SFPFpredictClosest) + '\n\n')

	outTex.write('\\vspace{3em}\n\n')

	## write R script ##
	Rscript.write('postscript(\'' + wordplotsDir + fullform + filesuffix + '.eps\')\n')
	Rscript.write('sf <- c(%s)\n' % (','.join([str(x) for x in logcountList])))
	Rscript.write('pf <- c(%s)\n' % (','.join([str(x) for x in logcountReversedList[::-1]])))
	Rscript.write('y_range <- range(sf,pf)\n')

	Rscript.write('plot(sf, type="o", pch=21, lty=1, ylim=y_range, axes=FALSE, ann=FALSE)\n')
	Rscript.write('lines(pf, type="o", pch=22, lty=2)\n')

	x_axis_label = ''
	for i in range(len(fullform)):
		if i < truncPoint:
			x_axis_label = x_axis_label + fullform[i].upper()
		else:
			x_axis_label = x_axis_label + fullform[i]

	Rscript.write('axis(1, at=1:%d, lab=c(%s))\n' % (len(fullform), ','.join(['"'+x+'"' for x in x_axis_label])))
	Rscript.write('axis(2, las=1)\n')

	Rscript.write('box()\n')

	Rscript.write('title(main="%s")\n' % (gold.replace('|','')))
	Rscript.write('title(ylab="log(count)")\n')

	Rscript.write('legend(2, y_range[2], c("R-complete count (RC)","L-complete count (LC)"), pch=21:22, lty=1:2)\n')
	Rscript.write('dev.off()\n\n')

	counter += 1


##########################################################
### evaluation
##########################################################

print

evaluationList = list()

SFevaluationList = list()
PFevaluationList = list()
SFPF_closest_evaluationList = list()
BinFootRLevaluationList = list()
OGBinFootevaluationList = list()

output.write('{0},{1},{2},{3},{4},{5},{6}\n'.format('word','trucpoint',
														'RC',
														'LC',
														'RCLC', 'BinaryFootRL', 'BinaryFootLR'))

for (gold, T, SF, PF, SFPF_closest, binfootRL, OGbinfoot) in zip(goldStandard, truncPointList, SFpredictList, PFpredictList, SFPFpredictClosestList, truncPointBinFootRLList, truncPointOGBinFootList):
	SFeval = SF - T
	PFeval = PF - T
	SFPFeval_closest = SFPF_closest - T
	BinFootRLEval = binfootRL - T
	OGBinFootEval = OGbinfoot - T

	SFevaluationList.append(SFeval)
	PFevaluationList.append(PFeval)
	SFPF_closest_evaluationList.append(SFPFeval_closest)
	BinFootRLevaluationList.append(BinFootRLEval)
	OGBinFootevaluationList.append(OGBinFootEval)

	output.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(gold, T,
														SFeval, 
														PFeval, 
														SFPFeval_closest, BinFootRLEval, OGBinFootEval))

lenGold = len(goldStandard)

## work from here, 2015-01-23, 3:30pm

output.write(',sum ->,{0},{1},{2},{3},{4}\n'.format(sum(SFevaluationList), 
													sum(PFevaluationList),
													sum(SFPF_closest_evaluationList), sum(BinFootRLevaluationList), sum(OGBinFootevaluationList)))

def sum_abs(L, plus=0):
	return sum([abs(x+plus) for x in L])

output.write(',abs. values ->,{0},{1},{2},{3},{4}\n'.format(sum_abs(SFevaluationList), 
															sum_abs(PFevaluationList), 
															sum_abs(SFPF_closest_evaluationList), sum_abs(BinFootRLevaluationList), sum_abs(OGBinFootevaluationList)))

def RMS(L):
	'''root mean square'''
	return math.sqrt(sum([x*x for x in L])/float(len(L)))

output.write(',RMS ->,{0},{1},{2},{3},{4}\n'.format(RMS(SFevaluationList), 
													RMS(PFevaluationList), 
													RMS(SFPF_closest_evaluationList),
													RMS(BinFootRLevaluationList),
													RMS(OGBinFootevaluationList)))

def proportion(List):
	return float(List.count(0))/len(List)

output.write('\n,correct proportion,{0},{1},{2},{3},{4}\n'.format(proportion(SFevaluationList),
																proportion(PFevaluationList),
																proportion(SFPF_closest_evaluationList),
																proportion(BinFootRLevaluationList), proportion(OGBinFootevaluationList)))


output.write('\n,mean,{0},{1},{2},{3},{4}\n'.format(np.mean(SFevaluationList),
													np.mean(PFevaluationList),
													np.mean(SFPF_closest_evaluationList),
													np.mean(BinFootRLevaluationList), np.mean(OGBinFootevaluationList)))


output.write('\n,std dev.,{0},{1},{2},{3},{4}\n'.format(np.std(SFevaluationList),
													np.std(PFevaluationList),
													np.std(SFPF_closest_evaluationList),
													np.std(BinFootRLevaluationList), np.std(OGBinFootevaluationList)))


############################################################
# close open file objects and compile .tex file

outTex.write('\\end{document}\n')
outTex.close()
output.close()
Rscript.close()

subprocess.call(('xelatex', outTex.name)) 

#------------------------------------------------------------------------------#
#	run R script for word plots

subprocess.call(('Rscript', Rscriptname))

#------------------------------------------------------------------------------#
#	write and run R script for error distribution plots

scriptstring = '''
# plot density curves of error distributions

data = read.csv('output%s.csv', header= TRUE)
data = subset(data, word != "")

linewidth = 2 # default is 1
density.adjust = 1.5 # default is 1

data.RC = data$RC
data.LC = data$LC
data.RCLC = data$RCLC
data.BinaryFootRL = data$BinaryFootRL
data.BinaryFootLR = data$BinaryFootLR

minX = min(c(data.RC, data.LC, data.RCLC, data.BinaryFootRL, data.BinaryFootLR))
maxX = max(c(data.RC, data.LC, data.RCLC, data.BinaryFootRL, data.BinaryFootLR))

minY = min(as.data.frame(table(data.RC))[2], as.data.frame(table(data.LC))[2], 
           as.data.frame(table(data.RCLC))[2], as.data.frame(table(data.BinaryFootRL))[2],
           as.data.frame(table(data.BinaryFootLR))[2])

maxY = max(as.data.frame(table(data.RC))[2], as.data.frame(table(data.LC))[2], 
           as.data.frame(table(data.RCLC))[2], as.data.frame(table(data.BinaryFootRL))[2],
           as.data.frame(table(data.BinaryFootLR))[2])

RC.density = density(data.RC, adjust=density.adjust)
LC.density = density(data.LC, adjust=density.adjust)
RCLC.density = density(data.RCLC, adjust=density.adjust)
binftRL.density = density(data.BinaryFootRL, adjust=density.adjust)
binftLR.density = density(data.BinaryFootLR, adjust=density.adjust)

minX.density = min(c(RC.density$x, LC.density$x, RCLC.density$x, binftRL.density$x, binftLR.density$x))
maxX.density = max(c(RC.density$x, LC.density$x, RCLC.density$x, binftRL.density$x, binftLR.density$x))
minY.density = min(c(RC.density$y, LC.density$y, RCLC.density$y, binftRL.density$y, binftLR.density$y))
maxY.density = max(c(RC.density$y, LC.density$y, RCLC.density$y, binftRL.density$y, binftLR.density$y))

postscript('%serror-distribution-density%s.eps')
plot(RC.density, lty=2, xlim=c(minX.density, maxX.density), ylim=c(minY.density, maxY.density), lwd=linewidth,
     main="", ylab="Density", xlab="Error")
lines(LC.density, lty=3, lwd=linewidth, ann=FALSE)
lines(RCLC.density, lty=1, lwd=linewidth, ann=FALSE)
lines(binftRL.density, lty=4, lwd=linewidth, ann=FALSE)
lines(binftLR.density, lty=5, lwd=linewidth, ann=FALSE)
legend(minX.density+1, maxY.density-0.05, c("RC","LC", "RC+LC", "BinFtRL", "BinFtLR"), lty=c(2,3,1,4,5), lwd=linewidth)

dev.off()


###################
# plot individual histograms

fillcolor = 'gray'
ylabname = 'Count'

postscript('%serror-distribution-histogram-RC%s.eps')
hist(data.RC, xlim=c(minX, maxX), ylim=c(minY, maxY), col=fillcolor, ylab=ylabname)
lines(x=RC.density$x, y=RC.density$y*100, lwd=linewidth)
dev.off()

postscript('%serror-distribution-histogram-LC%s.eps')
hist(data.LC, xlim=c(minX, maxX), ylim=c(minY, maxY), col=fillcolor, ylab=ylabname)
lines(x=LC.density$x, y=LC.density$y*100, lwd=linewidth)
dev.off()

postscript('%serror-distribution-histogram-RCLC%s.eps')
hist(data.RCLC, xlim=c(minX, maxX), ylim=c(minY, maxY), col=fillcolor, ylab=ylabname)
lines(x=RCLC.density$x, y=RCLC.density$y*100, lwd=linewidth)
dev.off()

postscript('%serror-distribution-histogram-BinFtRL%s.eps')
hist(data.BinaryFootRL, xlim=c(minX, maxX), ylim=c(minY, maxY), col=fillcolor, ylab=ylabname)
lines(x=binftRL.density$x, y=binftRL.density$y*100, lwd=linewidth)
dev.off()

postscript('%serror-distribution-histogram-BinFtLR%s.eps')
hist(data.BinaryFootLR, xlim=c(minX, maxX), ylim=c(minY, maxY), col=fillcolor, ylab=ylabname)
lines(x=binftLR.density$x, y=binftLR.density$y*100, lwd=linewidth)
dev.off()

''' % (filesuffix, errorplotsDir, filesuffix,
					errorplotsDir, filesuffix,
					errorplotsDir, filesuffix, 
					errorplotsDir, filesuffix, 
					errorplotsDir, filesuffix,
					errorplotsDir, filesuffix,
)

errorscriptname = errorplotsDir + 'plot_error_distributions.R'

with open(errorscriptname, 'w') as Rscriptfile:
	Rscriptfile.write(scriptstring)

subprocess.call(('Rscript', errorscriptname))


