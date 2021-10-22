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
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm


class BaseLazyCoordinates(ABC):

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    def __array__(self, dtype=None):
        return np.asanyarray(self[:], dtype=dtype)

    @abstractmethod
    def __iadd__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        new = copy.copy(self)
        new += other
        return new

    @abstractmethod
    def __isub__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        new = copy.copy(self)
        new -= other
        return new

    @abstractmethod
    def __imul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        new = copy.copy(self)
        new *= other
        return new

    @abstractmethod
    def __itruediv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        new = copy.copy(self)
        new /= other
        return new

    @abstractmethod
    def __imatmul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        new = copy.copy(self)
        new @= other
        return new

    @abstractmethod
    def __getitem__(self, item):
        return NotImplementedError

    def __len__(self):
        return self.shape[0]


class LazyCoordinates(BaseLazyCoordinates):
    def __init__(self, x, shift=None, scale=None, rot=None):
        self._x = x
        dim = self.shape[1]
        if shift is None:
            self._shift = np.zeros((1, dim))
        else:
            self._shift = np.array(shift)

        if scale is None:
            self._scale = 1
        else:
            self._scale = scale

        if rot is None:
            self._rot = np.eye(dim)
        else:
            self._rot = np.array(rot)

    def save_transform(self, filename):
        np.savez(filename, shift=self._shift, scale=self._scale, rot=self._rot)

    @property
    def shape(self):
        return self._x.shape

    def __copy__(self):
        return self.__class__(self._x, self._shift, self._scale, self._rot)

    def __iadd__(self, other):
        self._shift += other
        return self

    def __isub__(self, other):
        self._shift -= other
        return self

    def __imul__(self, other):
        self._scale *= other
        self._shift *= other
        return self

    def __itruediv__(self, other):
        self._scale /= other
        self._shift /= other
        return self

    def __imatmul__(self, other):
        self._rot = self._rot @ other
        self._shift = self._shift @ other
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            x = self._x[item[0]]
        else:
            x = self._x[item]
        x = x * self._scale
        x = x @ self._rot
        x += self._shift
        if isinstance(item, tuple):
            return x[(slice(None), *item[1:])]
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self._x)})'


class LazyFileCoordinates(LazyCoordinates):
    def __init__(self, filename, *args, **kwargs):
        with open(filename, 'rb') as f:
            major, minor = np.lib.format.read_magic(f)
            shape, *_ = np.lib.format.read_array_header_1_0(f)
        self._shape = shape
        super().__init__(filename, *args, **kwargs)

    @property
    def _x(self):
        return np.load(self.filename, mmap_mode='r')

    @_x.setter
    def _x(self, other):
        self.filename = other

    @property
    def shape(self):
        return self._shape

    def __copy__(self):
        return self.__class__(self.filename, self._shift, self._scale, self._rot)


class LazyMeanAggregatorCoordinates(BaseLazyCoordinates):
    def __init__(self, patches):
        self.patches = []
        for patch in patches:
            if isinstance(patch.coordinates, LazyMeanAggregatorCoordinates):
                # flatten hierarchy
                self.patches.extend(patch.coordinates.patches)
            else:
                self.patches.append(patch)
        self._dim = patches[0].shape[1]
        self._nodes = set()
        for patch in patches:
            self._nodes.update(patch.nodes)

        self._nodes = np.array(sorted(self._nodes))

    @property
    def shape(self):
        return len(self._nodes), self._dim

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item, *others = item
        else:
            others = ()
        nodes = self._nodes[item]
        out = self.get_coordinates(nodes)
        if others:
            return out[(slice(None), *others)]
        else:
            return out

    def __array__(self, dtype=None):
        # more efficient
        out = np.zeros(self.shape, dtype=dtype)
        return self.get_coordinates(self._nodes, out)

    def as_array(self, out=None):
        return self.get_coordinates(self._nodes, out)

    def get_coordinates(self, nodes, out=None):
        nodes = np.asanyarray(nodes)
        if out is None:
            out = np.zeros((len(nodes), self._dim))

        count = np.zeros((len(nodes),), dtype=np.int)
        for patch in tqdm(self.patches, position=0, leave=False, desc='get mean embedding'):
            index = [i for i, node in enumerate(nodes) if node in patch.index]
            count[index] += 1
            out[index] += patch.get_coordinates(nodes[index])
        out /= count[:, None]
        return out

    def __iadd__(self, other):
        for patch in self.patches:
            patch.coordinates += other
        return self

    def __isub__(self, other):
        for patch in self.patches:
            patch.coordinates -= other
        return self

    def __imul__(self, other):
        for patch in self.patches:
            patch.coordinates *= other
        return self

    def __itruediv__(self, other):
        for patch in self.patches:
            patch.coordinates /= other
        return self

    def __imatmul__(self, other):
        for patch in self.patches:
            patch.coordinates = patch.coordinates @ other
        return self

    def __copy__(self):
        new = self.__new__(type(self))
        new.patches = [copy.copy(patch) for patch in self.patches]
        new._nodes = self._nodes
        new._dim = self._dim
        return new

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.patches)})'
