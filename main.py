# -*- encoding: utf8 -*-

"""Modeling truncation in Brazilian Portuguese
Mike Pham and Jackson Lee
"""

from __future__ import print_function, division

import os
import math
import subprocess
import multiprocessing as mp
import argparse

import numpy as np
import pandas as pd
import seaborn as sns


def elbow_point(points):
    """Return the index in ``points`` for maximal curvature.

    Parameters
    ----------
    points : list of int

    Returns
    -------
    int
    """
    second_derivative_list = [points[x+1] + points[x-1] - 2 * points[x]
                              for x in range(1, len(points) - 1)]
    second_derivative_list_point_tuples = sorted(
        enumerate(second_derivative_list),
        key=lambda x: x[1], reverse=True)
    return second_derivative_list_point_tuples[0][0] + 2


def closest_intersection(points1, points2):
    """Return the index of intersection point between ``points1`` and
    ``points2``.

    points1[idx1] *   * points1[idx2]
                   \ /
                    *  <--- intersection point
                   / \
    points2[idx1] *   * points2[idx2]

    Parameters
    ----------
    points1, points2 : list of int

    Returns
    -------
    int
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


def sum_abs(number_list):
    """Return the sum of the absolute values of ``number_list``.

    Parameters
    ----------
    number_list : list of int

    Returns
    -------
    int
    """
    return sum([abs(x) for x in number_list])


def proportion(number_list):
    """Return the proportion of correct truncation prediction.

    Parameters
    ----------
    number_list : list of int

    Returns
    -------
    float
    """
    return number_list.count(0) / len(number_list)


def replace_digraphs(word_):
    """Return the given word processed for orthographic changes.

    Parameters
    ----------
    word_ : str

    Returns
    -------
    str
    """
    word_ = word_.lower()
    word_ = word_.replace('ch', 'S')
    word_ = word_.replace('lh', 'L')
    word_ = word_.replace('nh', 'N')
    word_ = word_.replace('ss', 's')
    word_ = word_.replace('rr', 'R')
    return word_


def jitter(number_list, std_dev=0.1):
    """Return the jittered ``number_list``.

    Each number is jittered by ``d``, where ``d`` is randomly drawn from
    a normal distribution of N(0, ``sigma``^2).

    Parameters
    ----------
    number_list : list of int
    std_dev : float

    Returns
    -------
    list of int
    """
    return [_shift(x, np.random.normal(loc=0, scale=std_dev))
            for x in number_list]


def _shift(x, y):
    """Return ``x`` shifted by ``y``.
    We flip a coin to decide if we want x - y or x + y.

    Parameters
    ----------
    x, y : float

    Returns
    -------
    float
    """
    coin = np.random.rand()
    if coin > 0.5:
        return x - y
    else:
        return x + y


def main():
    # -------------------------------------------------------------------------#
    # Command line interface

    lexicon_default = 'data/pt_br_full.txt'
    goldstandard_default = 'data/gold_standard.txt'

    parser = argparse.ArgumentParser(
        description='Modeling truncation in Brazilian Portuguese, '
                    'by Mike Pham and Jackson Lee')

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
    parser.add_argument('-d', '--digraphsfixed', action='store_const',
                        default=False, const=True,
                        help='Change orthographic digraphs into monographs '
                             '(default: False)')
    parser.add_argument('-x', '--lexicon', type=str, default=lexicon_default,
                        help='Lexicon file (default: %s)' % lexicon_default)
    parser.add_argument('-g', '--goldstandard', type=str,
                        default=goldstandard_default,
                        help='Gold standard file (default: %s)'
                             % goldstandard_default)

    args = parser.parse_args()

    use_token_frequency = args.freqtoken
    compile_latex = args.latex
    run_r_script = args.run_r_script
    digraphs_fixed = args.digraphsfixed
    lexicon_filename = args.lexicon
    goldstandard_filename = args.goldstandard

    # -------------------------------------------------------------------------#
    # make sure the directories for output files are present

    results_dir = 'results'  # directory for plots, CSV, etc.
    word_plots_dir = 'plots_for_words'

    if not os.path.isdir(word_plots_dir):
        os.makedirs(word_plots_dir)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    output_ready_stdout = '\toutput ready: {}'

    # -------------------------------------------------------------------------#
    # determine file suffix

    file_suffix = ''

    if use_token_frequency:
        file_suffix += '-tokenfreq'
    if digraphs_fixed:
        file_suffix += '-nodigraphs'

    # -------------------------------------------------------------------------#
    # read lexicon

    print("\nReading the lexicon file...")

    lex_freq_dict = dict()

    for line in open(lexicon_filename, encoding="utf8"):
        line = line.strip()
        if not line:
            continue
        if digraphs_fixed:
            line = replace_digraphs(line)
        line_split = line.split()
        word = line_split[0]
        try:
            freq = int(line_split[1])
        except (ValueError, IndexError):
            freq = 1
        lex_freq_dict[word] = int(freq)

    lex_keys = lex_freq_dict.keys()

    lex_log_freq_dict = dict()

    for word in lex_keys:
        lex_log_freq_dict[word] = math.log(lex_freq_dict[word], 10)

    # -------------------------------------------------------------------------#
    # read gold standard words

    test_words = list()
    true_trunc_points = list()
    binRL_trunc_points = list()
    binLR_trunc_points = list()

    for line in open(goldstandard_filename, encoding="utf8"):
        line = line.strip()
        if not line:
            continue

        if digraphs_fixed:
            line = replace_digraphs(line)

        annotated_word = line.split()[0]

        positions = dict()
        positions["$"] = annotated_word.index("$")  # binLR marked by $
        positions["#"] = annotated_word.index("#")  # binRL marked by #
        positions["|"] = annotated_word.index("|")  # gold standard by |

        for rank, (symbol, position) in enumerate(sorted(positions.items(),
                                                         key=lambda x: x[1])):
            positions[symbol] = position - rank

        binLR_trunc_points.append(positions["$"])
        binRL_trunc_points.append(positions["#"])
        true_trunc_points.append(positions["|"])

        test_words.append(
            annotated_word.replace("$", "").replace("#", "").replace("|", ""))

    # -------------------------------------------------------------------------#
    # compute right- and left-complete counts for each input word

    print("\nComputing right- and left-complete counts...")

    def compute_rc_lc_counts(input_word):
        print(input_word)
        rc_counts = list()
        lc_counts = list()

        # initialize reversed form
        test_word_reversed = input_word[::-1]

        trunc = ''
        trunc_reversed = ''

        # compute the counts of left- and right-completes
        for letter, letter_reversed in zip(input_word, test_word_reversed):
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

            rc_counts.append(count_)
            lc_counts = [count_reversed] + lc_counts

        return rc_counts, lc_counts

    p = mp.Pool(processes=mp.cpu_count())
    rc_lc_count_master_list = p.map(compute_rc_lc_counts, test_words)

    rc_count_master_list = list()
    lc_count_master_list = list()

    for rc_counts, lc_counts in rc_lc_count_master_list:
        rc_count_master_list.append(rc_counts)
        lc_count_master_list.append(lc_counts)

    # log-transform the right- and left-complete counts

    log_rc_master_list = list()
    log_lc_master_list = list()

    for rc_counts, lc_counts in zip(rc_count_master_list, lc_count_master_list):
        log_rc_list = list()
        log_lc_list = list()

        for rc_count, lc_count in zip(rc_counts, lc_counts):
            if rc_count > 0:
                log_rc = round(math.log(rc_count, 10), 2)
            else:
                log_rc = 0

            if lc_count > 0:
                log_lc = round(math.log(lc_count, 10), 2)
            else:
                log_lc = 0

            log_rc_list.append(log_rc)
            log_lc_list.append(log_lc)

        log_rc_master_list.append(log_rc_list)
        log_lc_master_list.append(log_lc_list)

    # -------------------------------------------------------------------------#
    # compute truncation points predicted by the Gries algorithm

    print("\nComputing the Gries truncation points...")

    def compute_gries_point(test_word):
        print(test_word)
        if test_word not in lex_keys:
            print('(%s NOT in the lexicon)' % test_word)
            return 0

        trunc = ''

        for letter in test_word:
            trunc = trunc + letter
            gries_dict = dict()

            for word in lex_keys:
                if word.startswith(trunc):
                    gries_dict[word] = lex_log_freq_dict[word]

            most_frequent_word = sorted(gries_dict.items(), key=lambda x: x[1],
                                        reverse=True)[0][0]
            if most_frequent_word == test_word:
                return len(trunc)

        print('(Warning: end of word reached for %s)' % test_word)
        return 0

    p = mp.Pool(processes=mp.cpu_count())
    gries_trunc_points = p.map(compute_gries_point, test_words)

    # -------------------------------------------------------------------------#
    # compute truncation points based on RC, LC, and RC+LC

    rc_trunc_points = list()
    lc_trunc_points = list()
    rclc_trunc_points = list()

    for log_rc_list, log_lc_list in zip(log_rc_master_list, log_lc_master_list):

        SF_trunc_point = elbow_point(log_rc_list)
        PF_trunc_point = elbow_point(log_lc_list)
        SFPF_trunc_point = closest_intersection(log_rc_list, log_lc_list)

        rc_trunc_points.append(SF_trunc_point)
        lc_trunc_points.append(PF_trunc_point)
        rclc_trunc_points.append(SFPF_trunc_point)

    # -------------------------------------------------------------------------#
    # write LaTeX output

    print("\nWriting LaTeX output...")

    out_tex_filename = os.path.join(results_dir,
                                    'individual_word_details%s.tex' %
                                    file_suffix)
    out_tex = open(out_tex_filename, mode="w", encoding="utf8")
    out_tex.write('\\documentclass[10pt]{article}\n')
    out_tex.write('\\usepackage{booktabs}\n')
    out_tex.write('\\usepackage{color}\n')
    out_tex.write('\\usepackage[letterpaper, margin=.2in]{geometry}\n')
    out_tex.write('\\usepackage{fontspec}\n')
    out_tex.write('\\setlength{\\parindent}{0em}\n')
    out_tex.write('\\begin{document}\n')

    for i, word in enumerate(test_words):
        counter = i + 1
        rc_counts = rc_count_master_list[i]
        lc_counts = lc_count_master_list[i]
        log_RC_list = log_rc_master_list[i]
        log_LC_list = log_lc_master_list[i]

        out_tex.write("{}\n\n".format(counter))
        out_tex.write("{}\n\n".format(word))

        out_tex.write("\\begin{tabular}{l|%s}\n" % ("l" * (len(word) + 1)))

        trunc_row = "trunc: & "
        RC_count_row = "R-complete count: & "
        log_RC_row = "log(RC) & "
        LC_count_row = "L-complete count: & "
        log_LC_row = "log(LC) & "

        for k, letter in enumerate(word):
            rc_count = rc_counts[k]
            lc_count = lc_counts[k]
            log_RC = log_RC_list[k]
            log_LC = log_LC_list[k]

            trunc_row += letter + " & "
            RC_count_row = "{}{} & ".format(RC_count_row, rc_count)
            LC_count_row = "{}{} & ".format(LC_count_row, lc_count)
            log_RC_row = "{}{} & ".format(log_RC_row, log_RC)
            log_LC_row = "{}{} & ".format(log_LC_row, log_LC)

        out_tex.write(trunc_row + "\\\\ \n")
        out_tex.write(RC_count_row + "\\\\ \n")
        out_tex.write(log_RC_row + "\\\\ \n")
        out_tex.write(LC_count_row + "\\\\ \n")
        out_tex.write(log_LC_row + "\\\\ \n")

        out_tex.write("\\end{tabular}\n\n")

        out_tex.write("true trunc point: {}\n\n".format(true_trunc_points[i]))
        out_tex.write("RC trunc point: {}\n\n".format(rc_trunc_points[i]))
        out_tex.write("LC trunc point: {}\n\n".format(lc_trunc_points[i]))
        out_tex.write("RC+LC trunc point: {}\n\n".format(rclc_trunc_points[i]))
        out_tex.write("binRL trunc point: {}\n\n".format(binRL_trunc_points[i]))
        out_tex.write("binLR trunc point: {}\n\n".format(binLR_trunc_points[i]))
        out_tex.write("Gries trunc point: {}\n\n".format(gries_trunc_points[i]))

        if counter % 4:
            out_tex.write("\\vspace{1em}\n\n")
        else:
            out_tex.write("\\newpage\n\n")

    out_tex.write("\\end{document}\n")
    out_tex.close()
    print(output_ready_stdout.format(out_tex_filename))

    # ------------------------------------------------------------------------ #
    # write R script for individual words' plots

    print("\nWriting R script for plotting individual words...")

    Rscriptname = os.path.join(results_dir, 'plot_words%s.R' % file_suffix)
    Rscript = open(Rscriptname, mode='w', encoding="utf8")

    for i, test_word in enumerate(test_words):
        log_RC_list = log_rc_master_list[i]
        log_LC_list = log_lc_master_list[i]
        true_trunc_point = true_trunc_points[i]

        Rscript.write('postscript(\'' + word_plots_dir + '/' + \
                      test_word + file_suffix + '.eps\')\n')
        Rscript.write('sf <- c(%s)\n' % (','.join(str(x) for x in log_RC_list)))
        Rscript.write('pf <- c(%s)\n' % (','.join(str(x) for x in log_LC_list)))
        Rscript.write('y_range <- range(sf,pf)\n')

        Rscript.write('plot(sf, type="o", pch=21, lty=1, ylim=y_range, ' +
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

        Rscript.write('legend(2, y_range[2], c("R-complete count (RC)", '
                      '"L-complete count (LC)"), pch=21:22, lty=1:2)\n')
        Rscript.write('dev.off()\n\n')

    Rscript.close()
    print(output_ready_stdout.format(Rscriptname))

    # -------------------------------------------------------------------------#
    # compile latex file and run R script

    devnull = open(os.devnull)

    if compile_latex:
        print("\nCompiling LaTeX file...")

        try:
            subprocess.call(('xelatex', '-output-directory=%s' % results_dir,
                             out_tex_filename),
                            stdout=devnull, stderr=subprocess.STDOUT)
        except (OSError, FileNotFoundError):
            print('The command "xelatex" is unavailable. No LaTeX compilation.')
        else:
            print(output_ready_stdout.format("PDF from " + out_tex_filename))

    if run_r_script:
        print("\nRunning R scripts...")

        try:
            subprocess.call(('Rscript', Rscriptname),
                            stdout=devnull, stderr=subprocess.STDOUT)
        except (OSError, FileNotFoundError):
            print('The command "Rscript" is unavailable. No R scripts are run.')
        else:
            print(output_ready_stdout.format("EPS's from " + Rscriptname))

    # ------------------------------------------------------------------------ #
    # CSV file for errors

    print("\nComputing errors...")

    SF_eval_list = list()
    PF_eval_list = list()
    SFPF_eval_list = list()
    binRL_eval_list = list()
    binLR_eval_list = list()
    gries_eval_list = list()

    for T, SF, PF, SFPF, binRL, binLR, gries in zip(true_trunc_points,
        rc_trunc_points, lc_trunc_points, rclc_trunc_points,
        binRL_trunc_points, binLR_trunc_points, gries_trunc_points):

        # "gries" is 0 when either test_word isn't in lexicon or
        # when the gries algorithm fails as the test_word still doesn't emerge
        # as the most frequent word even when the whole word is the truncated
        # stem
        # Setting "gries = T" means we (very generously) assume the gries
        # algorithm gives the *correct* prediction!
        if gries == 0:
            gries = T

        SF_eval = SF - T
        PF_eval = PF - T
        SFPF_eval = SFPF - T
        binRL_eval = binRL - T
        binLR_eval = binLR - T
        gries_eval = gries - T

        SF_eval_list.append(SF_eval)
        PF_eval_list.append(PF_eval)
        SFPF_eval_list.append(SFPF_eval)
        binRL_eval_list.append(binRL_eval)
        binLR_eval_list.append(binLR_eval)
        gries_eval_list.append(gries_eval)

    output_csv_filename = os.path.join(results_dir,
                                       'errors%s.csv' % file_suffix)

    with open(output_csv_filename, mode="w", encoding="utf8") as output:
        output.write('{0},{1},{2},{3},{4},{5},{6}\n'
                     .format('word', 'RC', 'LC', 'RCLC',
                             'BinRL', 'BinLR', 'Gries'))

        for (gold, SF_eval, PF_eval, SFPF_eval, binRL_eval, binLR_eval,
             gries_eval) in  zip(test_words, SF_eval_list, PF_eval_list,
                                 SFPF_eval_list, binRL_eval_list,
                                 binLR_eval_list, gries_eval_list):

            output.write('{0},{1},{2},{3},{4},{5},{6}\n'
                         .format(gold, SF_eval, PF_eval, SFPF_eval, binRL_eval,
                                 binLR_eval, gries_eval))

    print(output_ready_stdout.format(output_csv_filename))

    # ------------------------------------------------------------------------ #
    # Writing evaluation file

    print("\nEvaluating results...")

    stats_results_filename = os.path.join(results_dir,
                                          'evaluation%s.txt' % file_suffix)
    stats_results_file = open(stats_results_filename, mode="w", encoding="utf8")

    row_template = '{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
    row_float_template = ('{:<20}{:<15.3f}{:<15.3f}{:<15.3f}'
                          '{:<15.3f}{:<15.3f}{:<15.3f}\n')

    stats_results_file.write(row_template.format(
        '', 'RC', 'LC', 'RCLC', 'BinRL', 'BinLR', 'Gries'))

    stats_results_file.write(
        row_float_template.format(
            'sum', sum(SF_eval_list), sum(PF_eval_list), sum(SFPF_eval_list),
            sum(binRL_eval_list), sum(binLR_eval_list), sum(gries_eval_list)
        )
    )

    stats_results_file.write(
        row_float_template.format(
            'abs values', sum_abs(SF_eval_list), sum_abs(PF_eval_list),
            sum_abs(SFPF_eval_list), sum_abs(binRL_eval_list),
            sum_abs(binLR_eval_list), sum_abs(gries_eval_list)
        )
    )

    stats_results_file.write(
        row_float_template.format(
            'correct proportion', proportion(SF_eval_list),
            proportion(PF_eval_list), proportion(SFPF_eval_list),
            proportion(binRL_eval_list), proportion(binLR_eval_list),
            proportion(gries_eval_list)
        )
    )

    stats_results_file.write(
        row_float_template.format(
            'mean', np.mean(SF_eval_list), np.mean(PF_eval_list),
            np.mean(SFPF_eval_list), np.mean(binRL_eval_list),
            np.mean(binLR_eval_list), np.mean(gries_eval_list)
        )
    )

    stats_results_file.write(
        row_float_template.format(
            'std dev', np.std(SF_eval_list), np.std(PF_eval_list),
            np.std(SFPF_eval_list), np.std(binRL_eval_list),
            np.std(binLR_eval_list), np.std(gries_eval_list)
        )
    )

    stats_results_file.close()
    print(output_ready_stdout.format(stats_results_filename))

    # -------------------------------------------------------------------------#
    # Creating the boxplot.

    sns.set_style("whitegrid")

    models = ['RC', 'LC', 'RCLC', 'BinRL', 'BinLR', 'Gries']
    eval_data = [SF_eval_list, PF_eval_list, SFPF_eval_list,
                 binRL_eval_list, binLR_eval_list, gries_eval_list]
    eval_data_jittered = [jitter(list_) for list_ in eval_data]

    boxplot_data = pd.DataFrame({model: data
                                 for model, data in zip(models, eval_data)})
    boxplot_data_jittered = pd.DataFrame({model: data
                                          for model, data in
                                          zip(models, eval_data_jittered)})

    boxplot = sns.boxplot(data=boxplot_data, color='#c0c0c0')
    boxplot = sns.stripplot(data=boxplot_data_jittered, color='.3', jitter=True)
    boxplot.set(ylabel='Distance error', ylim=(-5, 9))
    boxplot_filename = os.path.join(results_dir,
                                    'error_distribution_boxplot%s.eps'
                                    % file_suffix)
    boxplot.get_figure().savefig(boxplot_filename, format='eps')
    print(output_ready_stdout.format(boxplot_filename))

    # -------------------------------------------------------------------------#

    print("\nAll done!")


if __name__ == '__main__':
    main()
