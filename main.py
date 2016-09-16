# Modeling truncation in Brazilian Portuguese
# Mike Pham, Jackson Lee

import os
import math
import subprocess
import numpy as np
import multiprocessing as mp
import argparse


def elbow_point(points):
    """
    takes a list of points
    and outputs the index of the list for maximal curvature
    """
    second_derivative_list = [points[x+1] + points[x-1] - 2 * points[x]
                              for x in range(1, len(points) - 1)]
    second_derivative_list_point_tuples = sorted(
        enumerate(second_derivative_list),
        key=lambda x: x[1], reverse=True)
    return second_derivative_list_point_tuples[0][0] + 2


def closest_intersection(points1, points2):
    """
    points1[idx1] *   * points1[idx2]
                   \ /
                    *  <--- intersection point
                   / \
    points2[idx1] *   * points2[idx2]
    """
    for (idx, (p1, p2)) in enumerate(zip(points1, points2)):
        if p2 > p1:
            idx1 = idx - 1
            idx2 = idx

            diff1 = abs(points1[idx1] - points2[idx1])
            diff2 = abs(points1[idx2] - points2[idx2])

            if diff1 <= diff2:
                return idx
            else:
                return idx + 1


def sum_abs(number_list, plus=0):
    return sum([abs(x + plus) for x in number_list])


def rms(number_list):
    """root mean square"""
    return math.sqrt(sum([x * x for x in number_list]) / len(number_list))


def proportion(number_list):
    return number_list.count(0) / len(number_list)

# -----------------------------------------------------------------------------#
# Command line interface for
# which orthographic representation system to use as input
# and whether to include technical terms

parser = argparse.ArgumentParser(description='Modeling truncation in Brazilian'
                                             ' Portuguese, '
                                             'by Mike Pham and Jackson Lee')

parser.add_argument('-t', '--tech', action='store_const',
                    default=False, const=True,
                    help='Include technical terms (default: False)')
parser.add_argument('-a', '--a_as_TS', action='store_const',
                    default=False, const=True,
                    help='Treat -a in truncated form as part of truncated stem'
                         ' (default: False)')
parser.add_argument('-c', '--c_orthography', action='store_const',
                    default=False, const=True,
                    help='Use the c-orthography (less phonetic) instead of the'
                         ' k-orthography (more phonetic) (default: False)')

args = parser.parse_args()
use_tech_terms = args.tech
use_c_orthography = args.c_orthography
treat_a_as_truncated_stem = args.a_as_TS

asuffix = input("Treat -a in TF as part of TS (maximal symbol overlap)? Y/N ")
asuffix = asuffix.lower()

if asuffix != "y":
    techterms = input("Include technical terms? Y/N ").lower()
    orthosys = input("Orthography system: k or c? ").lower()

# -----------------------------------------------------------------------------#
# make sure the directories for output files are present

word_plots_dir = 'plots_words'
error_plots_dir = 'plots_errors'

if not os.path.isdir(word_plots_dir):
    os.makedirs(word_plots_dir)

if not os.path.isdir(error_plots_dir):
    os.makedirs(error_plots_dir)

# -----------------------------------------------------------------------------#
# determine file suffix

file_suffix = ''

if treat_a_as_truncated_stem:
    file_suffix += '-a'
if not use_tech_terms:
    file_suffix += '-notech'
if use_c_orthography:
    file_suffix += '-c'

# -----------------------------------------------------------------------------#
# read lexicon

lexiconfile = 'pt_br_orthofix'
if use_c_orthography:
    lexiconfile += '-c.txt'
else:
    lexiconfile += '.txt'

lex_freq_dict = dict()

for line in open(os.path.join('data', lexiconfile)).readlines():
    line = line.strip()
    word, freq = line.split()
    lex_freq_dict[word] = int(freq)

lex_keys = lex_freq_dict.keys()

# -----------------------------------------------------------------------------#
#    read gold standard words

goldstandard_binRL_filename = os.path.join(
    'data', 'goldStandard_binRL_orthofix%s.txt' % file_suffix)
goldstandard_binLR_filename = os.path.join(
    'data', 'goldStandard_binLR_orthofix%s.txt' % file_suffix)

goldstandard_binRL_annotated = [x.strip().split("\t")[0].replace('#', '')
                                for x in open(goldstandard_binRL_filename)]
goldstandard_binLR_annotated = [x.strip().split("\t")[0].replace('#', '')
                                for x in open(goldstandard_binLR_filename)]

test_words = [x.replace('|', '').replace('$', '')
              for x in goldstandard_binRL_annotated]

goldstandard_list = [x.replace('$', '') for x in goldstandard_binRL_annotated]



goldstandard_binRL_list = [x.replace('|', '')
                           for x in goldstandard_binRL_annotated]
goldstandard_binLR_list = [x.replace('|', '')
                           for x in goldstandard_binLR_annotated]

# -----------------------------------------------------------------------------#
# extract truncation points

true_trunc_points = [x.index("|") for x in goldstandard_list]
binRL_trunc_points = [x.index("$") for x in goldstandard_binRL_list]
binLR_trunc_points = [x.index("$") for x in goldstandard_binLR_list]

# -----------------------------------------------------------------------------#
# compute right-completes and left-completes for each test word
# (right-completes = previously "successor frequencies = SF")
# (left-completes = previously "predecessor frequencies = PF")

print("computing right- and left-complete counts...")

def compute_SF_PF_counts(test_word):
    print(test_word)
    SF_counts = list()
    PF_counts = list()

    # initialize reversed form
    test_word_reversed = test_word[::-1]

    trunc = ''
    trunc_reversed = ''

    # compute the counts of left- and right-completes
    for letter, letter_reversed in zip(test_word, test_word_reversed):
        count_ = 0
        count_reversed = 0
        trunc = trunc + letter
        trunc_reversed = letter_reversed + trunc_reversed

        for word in lex_keys:

            # This counts the number of R-completes given a "prefix"
            if word.startswith(trunc):
                count_ += 1

            # This counts the number of L-completes given a "suffix"
            if word.endswith(trunc_reversed):
                count_reversed += 1

        SF_counts.append(count_)
        PF_counts = [count_reversed] + PF_counts

    return SF_counts, PF_counts

p = mp.Pool(processes=mp.cpu_count())
SF_PF_count_master_list = p.map(compute_SF_PF_counts, test_words)

SF_count_master_list = list()
PF_count_master_list = list()

for SF_counts, PF_counts in SF_PF_count_master_list:
    SF_count_master_list.append(SF_counts)
    PF_count_master_list.append(PF_counts)


# log-transform the right- and left-complete counts

log_SF_master_list = list()
log_PF_master_list = list()

for SF_counts, PF_counts in zip(SF_count_master_list, PF_count_master_list):
    log_SF_list = list()
    log_PF_list = list()

    for SF_count, PF_count in zip(SF_counts, PF_counts):
        if SF_count > 0:
            log_SF = round(math.log(SF_count, 10), 2)
        else:
            log_SF = 0

        if PF_count > 0:
            log_PF = round(math.log(PF_count, 10), 2)
        else:
            log_PF = 0

        log_SF_list.append(log_SF)
        log_PF_list.append(log_PF)

    log_SF_master_list.append(log_SF_list)
    log_PF_master_list.append(log_PF_list)


# -----------------------------------------------------------------------------#
#    compute truncation points based on SF, PF, and SF+PF

SF_trunc_points = list()
PF_trunc_points = list()
SFPF_trunc_points = list()

for log_SF_list, log_PF_list in zip(log_SF_master_list, log_PF_master_list):

    SF_trunc_point = elbow_point(log_SF_list)
    PF_trunc_point = elbow_point(log_PF_list)
    SFPF_trunc_point = closest_intersection(log_SF_list, log_PF_list)

    SF_trunc_points.append(SF_trunc_point)
    PF_trunc_points.append(PF_trunc_point)
    SFPF_trunc_points.append(SFPF_trunc_point)

# -----------------------------------------------------------------------------#
# write LaTeX output

print("writing LaTeX output...")

out_tex_filename = 'outlatex%s.tex' % file_suffix
out_tex = open(out_tex_filename, 'w')
out_tex.write('\\documentclass{article}\n')
out_tex.write('\\usepackage{booktabs}\n')
out_tex.write('\\usepackage{color}\n')
out_tex.write('\\usepackage[letterpaper, landscape, margin=.2in]{geometry}\n')
out_tex.write('\\usepackage{fontspec}\n')
out_tex.write('\\setlength{\\parindent}{0em}\n')
out_tex.write('\\begin{document}\n')

for i, word in enumerate(test_words):
    counter = i + 1
    SF_counts = SF_count_master_list[i]
    PF_counts = PF_count_master_list[i]
    log_RC_list = log_SF_master_list[i]
    log_LC_list = log_PF_master_list[i]

    out_tex.write("{}\n\n".format(counter))
    out_tex.write("{}\n\n".format(word))

    out_tex.write("\\begin{tabular}{l|%s}\n" % ("l" * (len(word) + 1)))

    trunc_row = "trunc: & "
    RC_count_row = "R-complete count: & "
    log_RC_row = "log(RC) & "
    LC_count_row = "L-complete count: & "
    log_LC_row = "log(LC) & "

    for k, letter in enumerate(word):
        SF_count = SF_counts[k]
        PF_count = PF_counts[k]
        log_RC = log_RC_list[k]
        log_LC = log_LC_list[k]

        trunc_row += letter + " & "
        RC_count_row = "{}{} & ".format(RC_count_row, SF_count)
        LC_count_row = "{}{} & ".format(LC_count_row, PF_count)
        log_RC_row = "{}{} & ".format(log_RC_row, log_RC)
        log_LC_row = "{}{} & ".format(log_LC_row, log_LC)

    out_tex.write(trunc_row + "\\\\ \n")
    out_tex.write(RC_count_row + "\\\\ \n")
    out_tex.write(log_RC_row + "\\\\ \n")
    out_tex.write(LC_count_row + "\\\\ \n")
    out_tex.write(log_LC_row + "\\\\ \n")

    out_tex.write("\\end{tabular}\n\n")

    out_tex.write("true trunc point: {}\n\n".format(true_trunc_points[i]))
    out_tex.write("RC trunc point: {}\n\n".format(SF_trunc_points[i]))
    out_tex.write("LC trunc point: {}\n\n".format(PF_trunc_points[i]))
    out_tex.write("RC+LC trunc point: {}\n\n".format(SFPF_trunc_points[i]))

    if counter % 4:
        out_tex.write("\\vspace{1em}\n\n")
    else:
        out_tex.write("\\newpage\n\n")

out_tex.write("\\end{document}\n")
out_tex.close()

#------------------------------------------------------------------------------#
# write R script for individual words' plots

print("writing R script for plotting individual words...")

Rscriptname = 'plot_words.R'
Rscript = open(Rscriptname, 'w')

for i, test_word in enumerate(test_words):
    log_RC_list = log_SF_master_list[i]
    log_LC_list = log_PF_master_list[i]
    true_trunc_point = true_trunc_points[i]

    Rscript.write('postscript(\'' + word_plots_dir + '/' + \
                  test_word + file_suffix + '.eps\')\n')
    Rscript.write('sf <- c(%s)\n' % (','.join([str(x) for x in log_RC_list])))
    Rscript.write('pf <- c(%s)\n' % (','.join([str(x) for x in log_LC_list])))
    Rscript.write('y_range <- range(sf,pf)\n')

    Rscript.write('plot(sf, type="o", pch=21, lty=1, ylim=y_range, ' + \
                  'axes=FALSE, ann=FALSE)\n')
    Rscript.write('lines(pf, type="o", pch=22, lty=2)\n')

    x_axis_label = ''
    for k in range(len(test_word)):
        if k < true_trunc_point:
            x_axis_label = x_axis_label + test_word[k].upper()
        else:
            x_axis_label = x_axis_label + test_word[k]

    Rscript.write('axis(1, at=1:%d, lab=c(%s))\n' % (
        len(test_word), ','.join(['"' + x + '"' for x in x_axis_label])))
    Rscript.write('axis(2, las=1)\n')

    Rscript.write('box()\n')

    Rscript.write('title(main="%s")\n' % test_word)
    Rscript.write('title(ylab="log(count)")\n')

    Rscript.write('legend(2, y_range[2], c("R-complete count (RC)", ' + \
        '"L-complete count (LC)"), pch=21:22, lty=1:2)\n')
    Rscript.write('dev.off()\n\n')

Rscript.close()

#------------------------------------------------------------------------------#
# evaluation

print("evaluation...")

SF_eval_list = list()
PF_eval_list = list()
SFPF_eval_list = list()
binRL_eval_list = list()
binLR_eval_list = list()

for T, SF, PF, SFPF, binRL, binLR in zip(true_trunc_points, SF_trunc_points,
    PF_trunc_points, SFPF_trunc_points, binRL_trunc_points, binLR_trunc_points):

    SF_eval = SF - T
    PF_eval = PF - T
    SFPF_eval = SFPF - T
    binRL_eval = binRL - T
    binLR_eval = binLR - T

    SF_eval_list.append(SF_eval)
    PF_eval_list.append(PF_eval)
    SFPF_eval_list.append(SFPF_eval)
    binRL_eval_list.append(binRL_eval)
    binLR_eval_list.append(binLR_eval)


output = open('output%s.csv' % file_suffix, 'w')
output.write('{0},,{1},{2},{3},{4},{5}\n'.format('word', 'RC', 'LC', 'RCLC',
                                                'BinRL', 'BinLR'))

for gold, SF_eval, PF_eval, SFPF_eval, binRL_eval, binLR_eval in \
    zip(goldstandard_list, SF_eval_list, PF_eval_list, SFPF_eval_list,
        binRL_eval_list, binLR_eval_list):

    output.write('{0},,{1},{2},{3},{4},{5}\n'.format(gold,
        SF_eval, PF_eval, SFPF_eval, binRL_eval, binLR_eval))

output.write(',sum ->,{0},{1},{2},{3},{4}\n'.format(
    sum(SF_eval_list), sum(PF_eval_list), sum(SFPF_eval_list),
    sum(binRL_eval_list), sum(binLR_eval_list)))

output.write(',abs values ->,{0},{1},{2},{3},{4}\n'.format(
    sum_abs(SF_eval_list), sum_abs(PF_eval_list), sum_abs(SFPF_eval_list),
    sum_abs(binRL_eval_list), sum_abs(binLR_eval_list)))

output.write(',RMS ->,{0},{1},{2},{3},{4}\n'.format(
    rms(SF_eval_list), rms(PF_eval_list), rms(SFPF_eval_list),
    rms(binRL_eval_list), rms(binLR_eval_list)))

output.write(',correct proportion,{0},{1},{2},{3},{4}\n'.format(
    proportion(SF_eval_list), proportion(PF_eval_list),
    proportion(SFPF_eval_list),
    proportion(binRL_eval_list), proportion(binLR_eval_list)))

output.write(',mean,{0},{1},{2},{3},{4}\n'.format(
    np.mean(SF_eval_list), np.mean(PF_eval_list), np.mean(SFPF_eval_list),
    np.mean(binRL_eval_list), np.mean(binLR_eval_list)))

output.write(',std dev,{0},{1},{2},{3},{4}\n'.format(
    np.std(SF_eval_list), np.std(PF_eval_list), np.std(SFPF_eval_list),
    np.std(binRL_eval_list), np.std(binLR_eval_list)))

output.close()

############################################################
# compile latex file and run R script

print("compiling LaTeX file...")
subprocess.call(('xelatex', out_tex_filename))

print("running R scripts...")
subprocess.call(('Rscript', Rscriptname))

#------------------------------------------------------------------------------#
#    write and run R script for error distribution plots

print("writing R script for plotting error distributions...")

scriptstring = '''
# plot density curves of error distributions

data = read.csv('output%s.csv', header= TRUE)
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
postscript('%s/error-distribution-density%s.eps')
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
postscript('%s/error-distribution-density-colored%s.eps')
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

postscript('%s/error-distribution-histogram-RC%s.eps')
hist(data.RC, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=RC.density$x, y=RC.density$y*100, lwd=linewidth)
dev.off()

postscript('%s/error-distribution-histogram-LC%s.eps')
hist(data.LC, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=LC.density$x, y=LC.density$y*100, lwd=linewidth)
dev.off()

postscript('%s/error-distribution-histogram-RCLC%s.eps')
hist(data.RCLC, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=RCLC.density$x, y=RCLC.density$y*100, lwd=linewidth)
dev.off()

postscript('%s/error-distribution-histogram-BinFtRL%s.eps')
hist(data.BinaryFootRL, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=binftRL.density$x, y=binftRL.density$y*100, lwd=linewidth)
dev.off()

postscript('%s/error-distribution-histogram-BinFtLR%s.eps')
hist(data.BinaryFootLR, xlim=c(minX, maxX), ylim=c(minY, maxY.overall),
    col=fillcolor, ylab=ylabname)
lines(x=binftLR.density$x, y=binftLR.density$y*100, lwd=linewidth)
dev.off()

''' % (file_suffix,
       error_plots_dir, file_suffix,
       error_plots_dir, file_suffix,
       error_plots_dir, file_suffix,
       error_plots_dir, file_suffix,
       error_plots_dir, file_suffix,
       error_plots_dir, file_suffix,
       error_plots_dir, file_suffix,
       )

errorscriptname = 'plot_error_distributions.R'

with open(errorscriptname, 'w') as Rscriptfile:
    Rscriptfile.write(scriptstring)

subprocess.call(('Rscript', errorscriptname))

print("all done!")
