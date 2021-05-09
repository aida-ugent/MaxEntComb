#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 08/07/2019

from __future__ import division
import numpy as np
from scipy.sparse import *
from abc import abstractmethod

# The structural priors implemented here assume unweighted graphs. Can be directed or undirected.


class WeightLinConstr:
    """
    Abstract class which defines the methods that all priors need to implement.
    In this case we assume we have O(n^2) space.
    """

    @abstractmethod
    def sc_mult(self, val):
        """ Multiply F with scalar val element-wise and return the result. """

    @abstractmethod
    def mat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F and mat. """

    @abstractmethod
    def sqmat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F^2 and mat. """

class CommonNeigh(WeightLinConstr):

    def __init__(self, A):
        if not issparse(A):
            A = csr_matrix(A)
        self.__F = self._compute_f(A)

    @staticmethod
    def _compute_f(A):
        F = A.dot(A.T)
        F.setdiag(0)
        F.eliminate_zeros()
        F.sort_indices()
        return F.multiply(1 / F.max())

    def sc_mult(self, val):
        """ Multiply F with scalar val element-wise and return the result. """
        return self.__F.multiply(val).A

    def mat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F and mat. """
        return (self.__F.multiply(mat)).sum()

    def sqmat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F^2 and mat. """
        aux = self.__F.multiply(self.__F)
        return (aux.multiply(mat)).sum()


class Jaccard(CommonNeigh):

    @staticmethod
    def _compute_f(A):
        inters = A.dot(A.T)
        degrees = inters.diagonal()
        inters.setdiag(0)
        inters.eliminate_zeros()
        (r, c) = inters.nonzero()
        union = degrees[r] + degrees[c] - inters.data
        F = csr_matrix((inters.data / union, (r, c)), shape=inters.shape)
        return F.multiply(1 / F.max())


class AdamicAdar(CommonNeigh):

    @staticmethod
    def _compute_f(A):
        degrees = A.sum(axis=1).A.ravel()
        weights = np.ma.divide(1.0, np.ma.log(degrees)).filled(0)
        F = (A.multiply(weights)).dot(A.T)
        F.setdiag(0)
        F.eliminate_zeros()
        return F.multiply(1 / F.max())


class ResourceAllocation(CommonNeigh):
    @staticmethod
    def _compute_f(A):
        degrees = A.sum(axis=1).A.ravel()
        weights = np.ma.divide(1.0, degrees).filled(0)
        F = (A.multiply(weights)).dot(A.T)
        F.setdiag(0)
        F.eliminate_zeros()
        return F.multiply(1 / F.max())


class WeightLinConstrOn:
    """
    Abstract class which defines the methods that all priors need to implement.
    In this case we assume we have only O(n) space, thus the F matrices can not be stored (compute them on the fly).
    """

    @abstractmethod
    def get_elem(self, i, j):
        """ Returns element in position i, j of F. """

    @abstractmethod
    def sc_row_mult(self, i, val):
        """ Returns the ith row of F multiplied by a scalar val. """

    @abstractmethod
    def dot(self, i, val):
        """ Returns the dot product between the ith row of F and dense vector val. """

    @abstractmethod
    def sqdot(self, i, val):
        """ Returns the dot product between the ith row of F^2 and dense vector val. """

    @abstractmethod
    def mat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F and mat. """


class CommonNeighOn(WeightLinConstrOn):

    def __init__(self, A):
        if not issparse(A):
            A = csr_matrix(A)
        A.setdiag(0)
        A.eliminate_zeros()
        self._A = A
        self._max = self._find_max()

    def _find_max(self):
        """ Can be more efficient if we take elements ordered by degree and stop when max>next_elem degree. """
        maxval = 0
        for i in range(self._A.shape[0]):
            aux = self._row_f(i)
            aux = aux.max()
            if aux > maxval:
                maxval = aux
        return maxval

    def _row_f(self, i, val=1.0):
        """ Computes the common neigh for row i multiplied by val and returns res as a row vector. """
        res = self._A.dot(val * self._A[i].T)
        res[i] = 0
        return res.T

    def get_elem(self, i, j):
        """ Returns element in position i, j of F. """
        if i == j:
            return 0.0
        else:
            return self._A[i].multiply(1 / self._max * self._A[j]).sum()

    def sc_row_mult(self, i, val):
        """ Returns the ith row of F multiplied by a scalar val. """
        if val == 0:
            return np.zeros(self._A.shape[0])
        else:
            return self._row_f(i, val/self._max).A.ravel()

    def dot(self, i, val):
        """ Returns the dot product between the ith row of F and dense vector val. """
        res = self._row_f(i, 1/self._max)
        return res.dot(val)[0]

    def sqdot(self, i, val):
        """ Returns the dot product between the ith row of F^2 and dense vector val. """
        res = self._row_f(i, 1/self._max)
        res = res.power(2)
        return res.dot(val)[0]

    def mat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F and mat. """
        matsum = 0.0
        for i in range(self._A.shape[0]):
            matsum += mat[i].multiply(self._row_f(i, 1.0 / self._max)).sum()
        return matsum


class JaccardOn(CommonNeighOn):

    def __init__(self, A):
        self._degrees = A.sum(axis=1).T.A.ravel()
        super(JaccardOn, self).__init__(A)

    def _row_f(self, i, val=1.0):
        """ Computes the jaccard score for row i multiplied by val and returns res as row vector. """
        inters = self._A.dot(self._A[i].T).T.A.ravel()
        inters[i] = 0
        unions = self._degrees[i] + self._degrees - inters
        res = csr_matrix(val * inters / unions)
        return res

    def get_elem(self, i, j):
        """ Returns element in position i, j of F. """
        if i == j:
            return 0.0
        else:
            inters = self._A[i].multiply(self._A[j]).sum()
            union = self._degrees[i] + self._degrees[j] - inters
            return (1 / self._max) * (inters / union)


class AdamicAdarOn(CommonNeighOn):

    def __init__(self, A):
        self._invdeglog = np.ma.divide(1.0, np.ma.log(A.sum(axis=1).A.ravel())).filled(0)
        super(AdamicAdarOn, self).__init__(A)

    def _row_f(self, i, val=1.0):
        """ Computes the adamic adar score for row i multiplied by val and returns res as row vector. """
        aux = self._A.multiply(self._A[i])
        aux = aux.multiply(self._invdeglog).sum(axis=1).T
        res = csr_matrix(val * aux)
        return res

    def get_elem(self, i, j):
        """ Returns element in position i, j of F. """
        if i == j:
            return 0.0
        else:
            aux = self._A[i].multiply(self._A[j])
            aux = aux.dot(self._invdeglog)
            return (1 / self._max) * aux


class ResourceAllocationOn(CommonNeighOn):

    def __init__(self, A):
        self._invdeg = np.ma.divide(1.0, A.sum(axis=1).A.ravel()).filled(0)
        super(ResourceAllocationOn, self).__init__(A)

    def _row_f(self, i, val=1.0):
        """ Computes the adamic adar score for row i multiplied by val and returns res as row vector. """
        aux = self._A.multiply(self._A[i])
        aux = aux.multiply(self._invdeg).sum(axis=1).T
        res = csr_matrix(val * aux)
        return res

    def get_elem(self, i, j):
        """ Returns element in position i, j of F. """
        if i == j:
            return 0.0
        else:
            aux = self._A[i].multiply(self._A[j])
            aux = aux.dot(self._invdeg)
            return (1 / self._max) * aux


class PrefAttach(WeightLinConstr, WeightLinConstrOn):

    def __init__(self, A):
        if not issparse(A):
            A = csr_matrix(A)
        self._compute_f(A)

    def _compute_f(self, A):
        # Get row and cod degrees
        rdeg = A.sum(axis=1).T.A.ravel()
        cdeg = A.sum(axis=0).A.ravel()
        self.__rdeg = rdeg.copy()
        self.__cdeg = cdeg.copy()
        # Get indexes of max per vector to compute max of matrix for normalization
        ramax1 = np.argmax(rdeg)
        camax1 = np.argmax(cdeg)
        if ramax1 != camax1:
            # If max is not one of diag elements (constr. to 0) we are done
            m = np.sqrt(rdeg[ramax1] * cdeg[camax1])
        else:
            # If max is one of diagonal elements, search for the non-diag max
            # Get the indexes of second larges elements of each vector
            rdeg[ramax1] = 0
            cdeg[camax1] = 0
            ramax2 = np.argmax(rdeg)
            camax2 = np.argmax(cdeg)
            # The largest combination max form one vect, second_max from the other, will give off-diag matrix max
            if self.__rdeg[ramax1] * self.__cdeg[camax2] >= self.__rdeg[ramax2] * self.__cdeg[camax1]:
                m = np.sqrt(self.__rdeg[ramax1] * self.__cdeg[camax2])
            else:
                m = np.sqrt(self.__rdeg[ramax2] * self.__cdeg[camax1])
        # Normalize the vectors by sqrt of max matrix element
        self.__rdeg = self.__rdeg/m        # Norm Row degrees
        self.__cdeg = self.__cdeg/m        # Norm Col degrees

    def get_mat(self):
        F = np.outer(self.__rdeg, self.__cdeg)
        np.fill_diagonal(F, 0)
        return F

    def get_row(self, i):
        r = self.__rdeg[i] * self.__cdeg
        r[i] = 0
        return r

    def get_col(self, j):
        c = self.__cdeg[j] * self.__rdeg
        c[j] = 0
        return c

    def get_elem(self, i, j):
        if i == j:
            return 0
        else:
            return self.__rdeg[i] * self.__cdeg[j]

    def sum(self):
        res = (np.sum(self.__rdeg) * np.sum(self.__cdeg)) - np.sum(self.__rdeg * self.__cdeg)
        return res

    def sqsum(self):
        rdeg_sq = np.square(self.__rdeg)
        cdeg_sq = np.square(self.__cdeg)
        res = np.sum(rdeg_sq) * np.sum(cdeg_sq) - np.sum(rdeg_sq * cdeg_sq)
        return res

    def rsum(self, i):
        return np.sum(self.get_row(i))

    def csum(self, j):
        return np.sum(self.get_col(j))

    def sc_row_mult(self, i, val):
        res = (val * self.__rdeg[i]) * self.__cdeg
        res[i] = 0
        return res

    def sc_mult(self, val):
        F = np.sign(val) * np.outer(self.__rdeg * np.sqrt(abs(val)), self.__cdeg * np.sqrt(abs(val)))
        np.fill_diagonal(F, 0)
        return F

    def dot(self, i, val):
        return np.dot(self.get_row(i), val)

    def sqdot(self, i, val):
        row_sq = np.square(self.get_row(i))
        return np.dot(row_sq, val)

    def mat_sum_mult(self, mat):
        if issparse(mat):
            r, c = mat.nonzero()
            aux = self.__rdeg[r] * self.__cdeg[c]
            aux[r == c] = 0
            return np.sum(aux * mat.data)
        else:
            return np.sum(self.get_mat() * mat)

    def sqmat_sum_mult(self, mat):
        # Mat should always be a dense matrix
        return np.sum(np.square(self.get_mat()) * mat)
