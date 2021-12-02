#  Copyright (c) 2021. Lucas G. S. Jeub
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import copy

import numpy as np
from local2global.lazy import BaseLazyCoordinates, LazyMeanAggregatorCoordinates, LazyFileCoordinates


class Patch:
    """
    Class for patch embedding
    """

    index = None
    """mapping of node index to patch coordinate index"""

    coordinates = None
    """patch embedding coordinates"""

    def __init__(self, nodes, coordinates=None):
        """
        Initialise a patch from a list of nodes and corresponding coordinates

        Args:
            nodes: Iterable of integer node indeces for patch
            coordinates: filename for coordinate file to be loaded on demand
        """
        self.nodes = np.asanyarray(nodes)
        self.index = {int(n): i for i, n in enumerate(nodes)}
        if coordinates is not None:
            if not isinstance(coordinates, BaseLazyCoordinates):
                self.coordinates = np.asanyarray(coordinates)
            else:
                self.coordinates = coordinates


    @property
    def shape(self):
        """
        shape of patch coordinates

        (`shape[0]` is the number of nodes in the patch
        and `shape[1]` is the embedding dimension)
        """
        return self.coordinates.shape

    def get_coordinates(self, nodes):
        """
        get coordinates for a list of nodes

        Args:
            nodes: Iterable of node indeces
        """
        return self.coordinates[[self.index[node] for node in nodes], :]

    def get_coordinate(self, node):
        """
        get coordinate for a single node

        Args:
            node: Integer node index
        """
        return self.coordinates[self.index[node], :]

    def __copy__(self):
        """return a copy of the patch"""
        instance = self.__new__(type(self))
        instance.__dict__.update(self.__dict__)
        instance.coordinates = copy.copy(self.coordinates)
        return instance


class MeanAggregatorPatch(Patch):
    def __init__(self, patches):
        coordinates = LazyMeanAggregatorCoordinates(patches)
        super().__init__(coordinates.nodes, coordinates)

    @property
    def patches(self):
        return self.coordinates.patches

    def get_coordinate(self, node):
        # avoid double index conversion
        return self.coordinates.get_coordinates([node])

    def get_coordinates(self, nodes):
        # avoid double index conversion
        return self.coordinates.get_coordinates(nodes)


class FilePatch(Patch):
    def __init__(self, nodes, filename, shift=None, scale=None, rot=None):
        super().__init__(nodes, LazyFileCoordinates(filename, shift=shift, scale=scale, rot=rot))