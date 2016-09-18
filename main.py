# Modeling truncation in Brazilian Portuguese
# Mike Pham and Jackson Lee

from __future__ import print_function
import os
from math import (sqrt, log)
import subprocess
import multiprocessing as mp
import argparse

import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns


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
    return sqrt(sum([x * x for x in number_list]) / len(number_list))


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
parser.add_argument('-f', '--freqtoken', action='store_const',
                    default=False, const=True,
                    help='Use token frequencies in lexicon '
                         '(default: False)')
parser.add_argument('-l', '--latex', action='store_const',
                    default=False, const=True,
                    help='Compile the output LaTeX file (default: False)')
parser.add_argument('-r', '--run_r_script', action='store_const',
                    default=False, const=True,
                    help='Run R script (default: False)')

args = parser.parse_args()
use_tech_terms = args.tech
use_c_orthography = args.c_orthography
treat_a_as_truncated_stem = args.a_as_TS
use_token_frequency = args.freqtoken
compile_latex = args.latex
run_r_script = args.run_r_script

# -----------------------------------------------------------------------------#
# make sure the directories for output files are present

results_dir = 'results'  # directory for plots, CSV, etc.
word_plots_dir = 'plots_for_words'

if not os.path.isdir(word_plots_dir):
    os.makedirs(word_plots_dir)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# -----------------------------------------------------------------------------#
# determine file suffix

file_suffix = ''

if treat_a_as_truncated_stem:
    file_suffix += '-a'
if use_tech_terms:
    file_suffix += '-tech'
if use_c_orthography:
    file_suffix += '-c'
if use_token_frequency:
    file_suffix += '-tokenfreq'

goldstandard_file_suffix = file_suffix.replace('-tokenfreq', '')

if treat_a_as_truncated_stem:
    file_suffix = '-a-tech'
    goldstandard_file_suffix = '-a-tech'
    print('NOTE: -a can be active only when -t (but nothing else) is as well.')

# -----------------------------------------------------------------------------#
# read lexicon

lexicon_file = 'pt_br_orthofix'

if use_c_orthography:
    lexicon_file += '-c.txt'
else:
    lexicon_file += '.txt'

lex_freq_dict = dict()

for line in open(os.path.join('data', lexicon_file)).readlines():
    line = line.strip()
    word, freq = line.split()
    lex_freq_dict[word] = int(freq)

lex_keys = lex_freq_dict.keys()

lex_log_freq_dict = dict()

for word in lex_keys:
    lex_log_freq_dict[word] = log(lex_freq_dict[word], 10)

# -----------------------------------------------------------------------------#
# read gold standard words

goldstandard_binRL_filename = os.path.join(
    'data', 'goldStandard_binRL_orthofix%s.txt' % goldstandard_file_suffix)
goldstandard_binLR_filename = os.path.join(
    'data', 'goldStandard_binLR_orthofix%s.txt' % goldstandard_file_suffix)

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

print("\nComputing right- and left-complete counts...")


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

            if use_token_frequency:
                word_weight = lex_log_freq_dict[word]
            else:
                word_weight = 1

            if use_token_frequency and word_weight == 0:
                word_weight = 0.1

            # This counts the number of R-completes given a "prefix"
            if word.startswith(trunc):
                count_ += word_weight

            # This counts the number of L-completes given a "suffix"
            if word.endswith(trunc_reversed):
                count_reversed += word_weight

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
            log_SF = round(log(SF_count, 10), 2)
        else:
            log_SF = 0

        if PF_count > 0:
            log_PF = round(log(PF_count, 10), 2)
        else:
            log_PF = 0

        log_SF_list.append(log_SF)
        log_PF_list.append(log_PF)

    log_SF_master_list.append(log_SF_list)
    log_PF_master_list.append(log_PF_list)

# -----------------------------------------------------------------------------#
# compute truncation points based on SF, PF, and SF+PF

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

print("\nWriting LaTeX output...")

out_tex_filename = os.path.join(results_dir,
                                'individual_word_details%s.tex' % (file_suffix))
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

# ---------------------------------------------------------------------------- #
# write R script for individual words' plots

print("\nWriting R script for plotting individual words...")

Rscriptname = os.path.join(results_dir, 'plot_words.R')
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

# -----------------------------------------------------------------------------#
# compile latex file and run R script

devnull = open(os.devnull, 'w')

if compile_latex:
    print("\nCompiling LaTeX file...")

    try:
        subprocess.call(('xelatex', '-output-directory=%s' % results_dir,
                         out_tex_filename),
                        stdout=devnull, stderr=subprocess.STDOUT)
    except (OSError, FileNotFoundError):
        print('The command "xelatex" is unavailable. No LaTeX compilation.')

if run_r_script:
    print("\nRunning R scripts...")

    try:
        subprocess.call(('Rscript', Rscriptname),
                        stdout=devnull, stderr=subprocess.STDOUT)
    except (OSError, FileNotFoundError):
        print('The command "Rscript" is unavailable. No R scripts are run.')

# ---------------------------------------------------------------------------- #
# CSV file for errors

print("\nComputing errors...")

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

output_csv_filename = os.path.join(results_dir, 'errors%s.csv' % file_suffix)

with open(output_csv_filename, 'w') as output:
    output.write('{0},{1},{2},{3},{4},{5}\n'.format('word', 'RC', 'LC', 'RCLC',
                                                    'BinRL', 'BinLR'))

    for gold, SF_eval, PF_eval, SFPF_eval, binRL_eval, binLR_eval in \
        zip(goldstandard_list, SF_eval_list, PF_eval_list, SFPF_eval_list,
            binRL_eval_list, binLR_eval_list):

        output.write('{0},{1},{2},{3},{4},{5}\n'.format(gold,
            SF_eval, PF_eval, SFPF_eval, binRL_eval, binLR_eval))

# ---------------------------------------------------------------------------- #
# Writing evaluation file

print("\nEvaluating results...")

stats_results_filename = os.path.join(results_dir,
                                      'evaluation%s.txt' % file_suffix)
stats_results_file = open(stats_results_filename, 'w')

row_template = '{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}\n'  # <20 to left-align
row_float_template = '{:<20}{:<15.2f}{:<15.2f}{:<15.2f}{:<15.2f}{:<15.2f}\n'

stats_results_file.write(row_template.format('', 'RC', 'LC', 'RCLC',
                                             'BinRL', 'BinLR'))

stats_results_file.write(row_float_template.format('sum',
    sum(SF_eval_list), sum(PF_eval_list), sum(SFPF_eval_list),
    sum(binRL_eval_list), sum(binLR_eval_list)))

stats_results_file.write(row_float_template.format('abs values',
    sum_abs(SF_eval_list), sum_abs(PF_eval_list), sum_abs(SFPF_eval_list),
    sum_abs(binRL_eval_list), sum_abs(binLR_eval_list)))

stats_results_file.write(row_float_template.format('RMS',
    rms(SF_eval_list), rms(PF_eval_list), rms(SFPF_eval_list),
    rms(binRL_eval_list), rms(binLR_eval_list)))

stats_results_file.write(row_float_template.format('correct proportion',
    proportion(SF_eval_list), proportion(PF_eval_list),
    proportion(SFPF_eval_list),
    proportion(binRL_eval_list), proportion(binLR_eval_list)))

stats_results_file.write(row_float_template.format('mean',
    np.mean(SF_eval_list), np.mean(PF_eval_list), np.mean(SFPF_eval_list),
    np.mean(binRL_eval_list), np.mean(binLR_eval_list)))

stats_results_file.write(row_float_template.format('std dev',
    np.std(SF_eval_list), np.std(PF_eval_list), np.std(SFPF_eval_list),
    np.std(binRL_eval_list), np.std(binLR_eval_list)))

anova = stats.f_oneway(SF_eval_list, PF_eval_list, SFPF_eval_list,
                       binRL_eval_list, binLR_eval_list)

print('\nOne-way ANOVA:\nF-value: {}\np-value: {}'.format(*anova),
      file=stats_results_file)

print('\nKolmogorov-Smirnov tests', file=stats_results_file)

for data, model in zip(
        [SF_eval_list, PF_eval_list, binRL_eval_list, binLR_eval_list],
        ['RC', 'LC', 'binRL', 'binLR']):
    ks = stats.ks_2samp(SFPF_eval_list, data)
    print('Models: RCLC and {} |'.format(model), end=' ',
          file=stats_results_file)
    print('KS statistic: {:.4f} p-value: {}'.format(*ks),
          file=stats_results_file)

stats_results_file.close()

# -----------------------------------------------------------------------------#
# boxplot

models = ['RC', 'LC', 'RCLC', 'BinRL', 'BinLR']
eval_data = [SF_eval_list, PF_eval_list, SFPF_eval_list,
             binRL_eval_list, binLR_eval_list]

boxplot_data = pd.DataFrame({model: data
                             for model, data in zip(models, eval_data)})

boxplot = sns.boxplot(data=boxplot_data)
boxplot.set(ylabel='Error', ylim=(-5, 9))
boxplot.get_figure().savefig(os.path.join(results_dir,
                                          'error_distribution_boxplot%s.pdf'
                                          % (file_suffix)))

# -----------------------------------------------------------------------------#

print("\nAll done!")
