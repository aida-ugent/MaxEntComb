#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 08/07/2019

from __future__ import division
from __future__ import print_function
import numpy as np
from weighted_lin_constr import *
from tqdm import tqdm
from sklearn.externals.joblib import Parallel, delayed


class MaxentCombined:

    def __init__(self, A, func, memory='quadratic'):
        """
        Initializes the maxent combined model

        Parameters
        ----------
        A : Scipy sparse matrix
            A sparse adjacency matrix representing a certain network.
        func : list
            A list containing one or a combination of {'CN', 'AA', 'A3', 'JC', 'PA'}.
        memory : string, optional
            A string indicating if the method should use O(n) - 'linear' or O(n*n) - 'quadratic' memory.
            Default is 'quadratic'.
        sampled_constr : bool, optional
            A bool indicating if sampling should be used to reduce the size of the constraint matrices to O(e).
        """
        self.__n = A.shape[0]
        self.__2n = 2 * self.__n
        self.__nfuncs = len(func)
        self.__memory = memory
        self.__F = self._select_f(A, func)          # array with as many priors as __nfuncs
        self.__P = None
        self.__x = None
        self.__cs = self._get_sums(A)

    def _select_f(self, A, func):
        print("Initializing MaxentCombined with the following priors: {}".format(func))
        fs = list()
        for fi in range(self.__nfuncs):
            fs.append(self._select(A, func[fi]))
        return fs

    def _select(self, A, func):
        if func == 'CN':
            if self.__memory == 'quadratic':
                return CommonNeigh(A)
            else:
                return CommonNeighOn(A)
        elif func == 'AA':
            if self.__memory == 'quadratic':
                return AdamicAdar(A)
            else:
                return AdamicAdarOn(A)
        elif func == 'JC':
            if self.__memory == 'quadratic':
                return Jaccard(A)
            else:
                return JaccardOn(A)
        elif func == 'RA':
            if self.__memory == 'quadratic':
                return ResourceAllocation(A)
            else:
                return ResourceAllocationOn(A)
        elif func == 'PA':
            return PrefAttach(A)
        else:
            raise ValueError("Method not implemented.")

    def predict(self, E):
        """
        Returns the link predictions for the given set of (src, dst) pairs.

        Parameters
        ----------
        E : iterable
            An iterable of (src, dst) pairs.

        Returns
        -------
        scores : list
            The probabilities of linking each of the (src, dst) pairs, in the same order as E.

        Raises
        ------
        AttributeError
            If the method has not been fitted.
        """
        if self.__P is not None:
            scores = list()
            for src, dst in E:
                scores.append(self.__P[src, dst])
        else:
            if self.__x is not None:
                scores = list()
                for src, dst in E:
                    scores.append(self._get_elem_posterior(src, dst))
            else:
                raise AttributeError("Maxent Combined has not been fitted. Use <class>.fit()")
        return scores

    def get_posterior(self):
        """
        Returns the posterior probability matrix. This could fail if the linear memory constraint was used and the
        full matrix does not fit in memory.

        Returns
        -------
        P : ndarray
            The posterior probability matrix.

        Raises
        ------
        AttributeError
            If the method has not been fitted.
        """
        if self.__P is not None:
            return self.__P
        else:
            if self.__x is not None:
                return self._get_posterior()
            else:
                raise AttributeError("Maxent Combined has not been fitted. Use <class>.fit()")

    def fit(self, optimizer='grad_desc', lr=1.0, max_iter=100, tol=0.001, verbose=False):
        """
        Fits the Maxent Combined model.

        Parameters
        ----------
        optimizer : basestring, optional
            A string indicating the optimizer to use. For now only `grad_desc` is available. Default is `grad_desc`.
        lr : float, optional
            The learning rate. Default is 1.0.
        max_iter : int, optional
            The maximum number of iteration the optimization will be run for. Default is 100.
        tol : float, optional
            Triggers early stop if gradient norm is below this value.
        verbose : bool
            If True information is printed in each iteration of the optimization process.

        Raises
        ------
        ValueError
            If the the memory constraint or optimizer values are not correct.
        """
        # Initial condition
        x = np.zeros(self.__2n + self.__nfuncs)

        # Compute
        if optimizer == 'grad_desc':
            if self.__memory == 'quadratic':
                self.__P = self._optimizer_gd_quadratic(x, lr, max_iter, tol, verbose)
            elif self.__memory == 'linear':
                self.__x = self._optimizer_gd_linear(x, lr, max_iter, tol, verbose)
            else:
                raise ValueError('Incorrect memory constraint. Options are `quadratic` and `linear`')
        elif optimizer == 'newton':
            if self.__memory == 'quadratic':
                self.__P = self._optimizer_newton_quadratic(x, lr, max_iter, tol, verbose)
            elif self.__memory == 'linear':
                self.__x = self._optimizer_newton_linear(x, lr, max_iter, tol, verbose)
            else:
                raise ValueError('Incorrect memory constraint. Options are `quadratic` and `linear`')
        else:
            raise ValueError('Optimizer {:s} is not implemented.'.format(optimizer))

    def _optimizer_newton_quadratic(self, x, alpha_init=1.0, max_iter=100, tol=0.001, verbose=False):
        """
        Uses hessian information o speed up convergence.
        """

        print("Starting optimization...")
        alpha = alpha_init

        # Initialize posterior prob. matrix ( P = exp(sum(lbda_l*f_l)) )
        P = np.ones((self.__n, self.__n))

        # Initialize the objective
        obj = (self.__n ** 2 * 0.6931 - self.__n * 0.6931) - np.dot(x, self.__cs)
        obj = obj / self.__n
        # Iterate
        for i in tqdm(range(max_iter)):

            # Reset alpha every few iterations
            if i % 10 == 0:
                alpha = alpha_init

            # Compute gradient
            P1 = np.divide(P, (1 + P))
            np.fill_diagonal(P1, 0)
            grad = self._get_sums(P1) - self.__cs
            grad = grad / self.__n
            P2 = np.divide(P1, (1 + P))
            np.fill_diagonal(P2, 0)
            hess = self._get_sums(P2, f_pow=True)
            hess = hess / self.__n
            delta = grad / (hess + 0.00000001)

            # Greedy search for best alpha
            while True:
                x_test = x - alpha * delta
                P_test = np.array([x_test[:self.__n]]).T + x_test[self.__n:self.__2n]
                for fi in range(self.__nfuncs):
                    P_test += self.__F[fi].sc_mult(x_test[self.__2n + fi])
                P_test = np.exp(P_test)
                obj_test = self._compute_obj(P_test, x_test)
                obj_test = obj_test / self.__n
                if obj_test <= obj - (0.0001 * alpha * np.dot(delta, grad)):
                    x = x_test
                    P = P_test
                    obj = obj_test
                    break
                alpha = alpha / 2.0

            # Show the norms of the gradient and objective
            if verbose:
                print("\nGradient norm: {} in iter {}".format(np.linalg.norm(grad), i))
                print("Objective: {} in iter {}".format(obj, i))

            if np.linalg.norm(grad, np.inf) < tol:
                break

        print("Gradient norm: {}".format(np.linalg.norm(grad)))
        print("Objective: {}".format(obj))

        # Return the posterior probability matrix
        P = np.divide(P, (1 + P))
        np.fill_diagonal(P, 0)
        return P

    def _optimizer_newton_linear(self, x, alpha_init=1.0, max_iter=100, tol=0.001, verbose=False):
        """
        Uses hessian information to compute the lambdas. Will only require O(n) memory.
        It's slower than the quadratic memory version and requires the posterior probabilities to be computed
        at prediction time from the lbdas.
        """
        print("Starting optimization using O(n) memory...")
        alpha = alpha_init

        obj, gsums, hsums = self._opt_lin_sums(x, hessian=True)

        # Iterate
        for i in tqdm(range(max_iter)):

            # Reset alpha every few iterations
            if i % 10 == 0:
                alpha = alpha_init

            # Compute new gradient
            grad = gsums - (self.__cs / self.__n)
            delta = grad / (hsums + 0.00000001)
            graddot = np.dot(delta, grad)

            # Greedy search for best alpha
            while True:
                x_test = x - alpha * delta
                obj_test, gsums_test, hsums_test = self._opt_lin_sums(x_test, hessian=True)
                if obj_test <= obj - (0.0001 * alpha * graddot):
                    x = x_test
                    gsums = gsums_test
                    hsums = hsums_test
                    obj = obj_test
                    break
                alpha = alpha / 2.0

            # Show the norms of the gradient and objective
            if verbose:
                print("\nGradient norm: {} in iter {}".format(np.linalg.norm(grad), i))
                print("Objective: {} in iter {}".format(obj, i))

            if np.linalg.norm(grad, np.inf) < tol:
                break

        print("Gradient norm: {}".format(np.linalg.norm(grad)))
        print("Objective: {}".format(obj))
        return x

    def _optimizer_gd_linear(self, x, alpha_init=1.0, max_iter=100, tol=0.001, verbose=False):
        """
        Computes the lambdas using gradient descend. Will only require O(n) memory.
        It's slower than the quadratic memory version and requires the posterior probabilities to be computed
        at prediction time from the lbdas.
        """
        print("Starting optimization using O(n) memory...")
        alpha = alpha_init
        obj, sums = self._opt_lin_sums(x)

        # Iterate
        for i in tqdm(range(max_iter)):

            # Reset alpha every few iterations
            if i % 10 == 0:
                alpha = alpha_init

            # Compute new gradient
            grad = sums - (self.__cs / self.__n)
            graddot = np.dot(grad, grad)

            # Greedy search for best alpha
            while True:
                x_test = x - alpha * grad
                obj_test, sums_test = self._opt_lin_sums(x_test)
                if obj_test <= obj - (0.0001 * alpha * graddot):
                    x = x_test
                    sums = sums_test
                    obj = obj_test
                    break
                alpha = alpha / 2.0

            # Show the norms of the gradient and objective
            if verbose:
                print("\nGradient norm: {} in iter {}".format(np.linalg.norm(grad), i))
                print("Objective: {} in iter {}".format(obj, i))

            if np.linalg.norm(grad, np.inf) < tol:
                break

        print("Gradient norm: {}".format(np.linalg.norm(grad)))
        print("Objective: {}".format(obj))
        return x

    def _opt_lin_sums(self, x_test, hessian=False):
        """
        Computes the sums over rows, cols, total elems and objective of a posterior prob matrix obtained from
        the given lambdas.
        """
        obj = 0
        sums = np.zeros(self.__2n + self.__nfuncs)
        if hessian:
            hsums = np.zeros(self.__2n + self.__nfuncs)

        for i in range(self.__n):
            p = x_test[i] + x_test[self.__n:self.__2n]
            for fi in range(self.__nfuncs):
                p += self.__F[fi].sc_row_mult(i, x_test[self.__2n + fi])
            aux = np.exp(p)
            aux[i] = 0
            obj += np.sum(np.log(1 + aux))
            aux2 = aux / (1 + aux)
            if hessian:
                aux3 = aux2 / (1 + aux)
                hsums[i] = np.sum(aux3)
                hsums[self.__n:self.__2n] += aux3
                for fi in range(self.__nfuncs):
                    hsums[self.__2n + fi] += self.__F[fi].sqdot(i, aux3)
            sums[i] = np.sum(aux2)
            sums[self.__n:self.__2n] += aux2
            for fi in range(self.__nfuncs):
                sums[self.__2n + fi] += self.__F[fi].dot(i, aux2)

        # Compute objective
        obj = obj - np.dot(x_test, self.__cs)
        if hessian:
            return obj / self.__n, sums / self.__n, hsums / self.__n
        else:
            return obj / self.__n, sums / self.__n

    def _optimizer_gd_quadratic(self, x, alpha_init=1.0, max_iter=100, tol=0.001, verbose=False):
        """
        Computes the posterior probability matrix using gradient descend. Will require memory in O(n^2).
        If the graph fits in memory this implementation is faster than the linear memory one.
        """

        print("Starting optimization...")
        alpha = alpha_init

        # Initialize posterior prob. matrix ( P = exp(sum(lbda_l*f_l)) )
        P = np.ones((self.__n, self.__n))

        # Initialize the objective
        obj = (self.__n**2 * 0.6931 - self.__n * 0.6931) - np.dot(x, self.__cs)
        obj = obj / self.__n
        # Iterate
        for i in tqdm(range(max_iter)):

            # Reset alpha every few iterations
            if i % 10 == 0:
                alpha = alpha_init

            # Compute gradient
            P = np.divide(P, (1 + P))
            np.fill_diagonal(P, 0)
            grad = self._get_sums(P) - self.__cs
            grad = grad / self.__n

            # Greedy search for best alpha
            while True:
                x_test = x - alpha * grad
                P_test = np.array([x_test[:self.__n]]).T + x_test[self.__n:self.__2n]
                for fi in range(self.__nfuncs):
                    P_test += self.__F[fi].sc_mult(x_test[self.__2n + fi])
                P_test = np.exp(P_test)
                obj_test = self._compute_obj(P_test, x_test)
                obj_test = obj_test / self.__n
                if obj_test <= obj - (0.0001 * alpha * np.dot(grad, grad)):
                    x = x_test
                    P = P_test
                    obj = obj_test
                    break
                alpha = alpha / 2.0

            # Show the norms of the gradient and objective
            if verbose:
                print("\nGradient norm: {} in iter {}".format(np.linalg.norm(grad), i))
                print("Objective: {} in iter {}".format(obj, i))

            if np.linalg.norm(grad, np.inf) < tol:
                break

        print("Gradient norm: {}".format(np.linalg.norm(grad)))
        print("Objective: {}".format(obj))

        # Return the posterior probability matrix
        P = np.divide(P, (1 + P))
        np.fill_diagonal(P, 0)
        return P

    def _compute_obj(self, P, x):
        """ Computes the objective value for the given P and x as sum(log(1 + P)) - dot(x,cs). """
        obj_test = np.log(1 + P)
        np.fill_diagonal(obj_test, 0)
        return np.sum(obj_test) - np.dot(x, self.__cs)

    def _get_sums(self, P, f_pow=False):
        """ Returns the sums over cols and rows of P as well as sum over all elems of P * F in a single array. """
        aux = np.zeros(self.__2n + self.__nfuncs)
        aux[:self.__n] = P.sum(axis=1).T                                    # Row sums
        aux[self.__n:self.__2n] = P.sum(axis=0)                             # Col sums
        if f_pow:
            for fi in range(self.__nfuncs):
                aux[self.__2n + fi] = self.__F[fi].sqmat_sum_mult(P)        # Full sum
        else:
            for fi in range(self.__nfuncs):
                aux[self.__2n + fi] = self.__F[fi].mat_sum_mult(P)          # Full sum
        return aux

    def _get_posterior(self):
        """ Returns the full posterior probability matrix computed from the lambdas. """
        P = np.array([self.__x[:self.__n]]).T + self.__x[self.__n:self.__2n]
        for fi in range(self.__nfuncs):
            P += self.__F[fi].sc_mult(self.__x[self.__2n + fi])
        P = np.exp(P)
        P = np.divide(P, (1 + P))
        np.fill_diagonal(P, 0)
        return P

    def _get_elem_posterior(self, src, dst):
        """ Returns the probability of linking a (src, dst) pair. """
        if src == dst:
            return 0
        else:
            p = self.__x[src] + self.__x[self.__n + dst]
            for fi in range(self.__nfuncs):
                p += self.__F[fi].get_elem(src, dst) * self.__x[self.__2n + fi]
            p = np.exp(p)
            return p / (1 + p)
