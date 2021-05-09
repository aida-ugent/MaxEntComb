#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 29/01/2019

from __future__ import absolute_import
import time
import argparse
import numpy as np
import networkx as nx
from maxentcomb.maxent_comb import MaxentCombined


def parse_args():
    """ Parses Maxent Combined arguments. """

    parser = argparse.ArgumentParser(description="Run Maxent Combined.")

    parser.add_argument('--inputgraph', nargs='?',
                        default='sample/network.edgelist',
                        help='Input graph path')
    
    parser.add_argument('--output', nargs='?',
                        default='network.emb',
                        help='Path where the embeddings will be stored.')

    parser.add_argument('--tr_e', nargs='?', default=None,
                        help='Path of the input train edges. Default None (in this case returns embeddings)')

    parser.add_argument('--tr_pred', nargs='?', default='tr_pred.csv',
                        help='Path where the train predictions will be stored.')

    parser.add_argument('--te_e', nargs='?', default=None,
                        help='Path of the input test edges. Default None.')

    parser.add_argument('--te_pred', nargs='?', default='te_pred.csv',
                        help='Path where the test predictions will be stored.')

    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='Learning rate. Default is 0.1.')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs. Default is 100.')

    parser.add_argument('--grad_min_norm', type=float, default=0.0001,
                        help='Early stop if grad norm is below this value. Default is 0.0001.')

    parser.add_argument('--optimizer', default='newton',
                        help='Optimizer to use. Options are `newton` and `grad_desc`. Default is `newton`.')

    parser.add_argument('--memory', default='quadratic',
                        help='The constraints on the memory to use. Options are `quadratic` and `linear`. '
                             'Quadratic is considerably faster but used O(n^2) memory. Default is `quadratic`.')

    parser.add_argument('--prior', nargs='+', default=['CN'],
                        help='The prior to use. Options are `CN`, `AA`, `A3`, `JC`, `PA` or list combining them. '
                             'Default is [`CN`].')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used in the input edgelist. Output will use the same one.')

    parser.add_argument('--verbose', action='store_true',
                        help='Determines the verbosity level of the output.')
    parser.set_defaults(verbose=False)

    parser.add_argument('--dimension', type=int, default=1,
                        help='Not used, but required by EvalNE to run evaluation.')

    return parser.parse_args()


def main_helper(args):
    """ Main of Maxent Combined. """

    # Load edgelist
    E = np.loadtxt(args.inputgraph, delimiter=args.delimiter, dtype=int)

    # Create a graph
    G = nx.Graph()

    # We make sure the graph is unweighted
    G.add_edges_from(E[:, :2])

    # Get adj matrix of the graph
    tr_A = nx.adjacency_matrix(G, weight=None)
    
    # Initialize and fit the MaxentCombined model
    start = time.time()
    mc = MaxentCombined(tr_A, args.prior, args.memory)
    mc.fit(optimizer=args.optimizer, lr=args.learning_rate, max_iter=args.epochs, tol=args.grad_min_norm,
           verbose=args.verbose)
    end = time.time() - start
    print("Execution time: {}".format(end))

    start = time.time()
    # Read the train edges and run predictions
    if args.tr_e is not None:
        train_edges = np.loadtxt(args.tr_e, delimiter=args.delimiter, dtype=int)
        pred_tr = mc.predict(train_edges)
        np.savetxt(args.tr_pred, pred_tr, delimiter=args.delimiter)

        # Read the test edges and run predictions
        if args.te_e is not None:
            test_edges = np.loadtxt(args.te_e, delimiter=args.delimiter, dtype=int)
            pred_te = mc.predict(test_edges)
            np.savetxt(args.te_pred, pred_te, delimiter=args.delimiter)

    # If no edge lists provided to predict links, then just store the posterior
    else:
        np.savetxt(args.output, mc.get_posterior(), delimiter=args.delimiter)
    print('Prediction time: {}'.format(time.time()-start))


def main():
    args = parse_args()
    main_helper(args)


if __name__ == "__main__":
    main()
