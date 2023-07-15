"""
Experimental implementation of xarray custom index

https://github.com/corteva/rioxarray/issues/588
https://github.com/xarray-contrib/xvec/blob/main/xvec/index.py
"""
from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping
from typing import Any

import pandas as pd
from rasterio.crs import CRS
from xarray import Variable, get_options
from xarray.core.indexes import Index, PandasIndex, get_indexer_nd
from xarray.core.indexing import IndexSelResult, merge_sel_results

from rioxarray._options import CRS_INDEX_ERROR, get_option


# From XVEC
def _format_crs(crs: CRS, max_width: int = 50) -> str:
    if crs is not None:
        srs = crs.to_string()
    else:
        srs = "None"

    return srs if len(srs) <= max_width else " ".join([srs[:max_width], "..."])


class CRSIndex(Index):
    """Coordinate-Refernce System aware, Xarray compatible 2D index for rasters.
    A 'MetaIndex' that adds CRS information to 2 related spatial dimensions (x,y)
    The CRS defines coordinate system wherein the (x,y) points reside
    """

    def __init__(self, indexes: [PandasIndex, PandasIndex], crs: CRS):
        self._indexes = indexes
        self._crs = crs

    @property
    def crs(self) -> CRS | None:
        """Returns the coordinate reference system of the index as a
        :class:`rasterio.crs.CRS` object.
        """
        return self._crs

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ):

        xy_indexes = {
            k: PandasIndex.from_variables({k: v}, options=options)
            for k, v in variables.items()
            if k in ["x", "y"]
        }

        return cls(xy_indexes, crs=options.get("crs"))

    def create_variables(self, variables=None):
        idx_variables = {}

        for index in self._indexes.values():
            idx_variables.update(index.create_variables(variables))

        return idx_variables

    def sel(self, labels, **kwargs) -> IndexSelResult:
        print(self, labels, kwargs)
        results = []
        for k, index in self._indexes.items():
            if k in labels:
                results.append(index.sel({k: labels[k]}, **kwargs))

        result = merge_sel_results(results)
        return result

    def isel(self, indexers):
        # Do a CRS check
        # print(indexers)
        results = {}
        for k, index in self._indexes.items():
            if k in indexers:
                result_idx = index.isel(indexers)
                if result_idx is not None:
                    results[k] = result_idx
            else:
                results[k] = index

        return type(self)(results, crs=self.crs)

    def to_pandas_index(self) -> pd.Index:
        return self._index.index

    def _repr_inline_(self, max_width: int):
        # TODO: remove when fixed in XArray, Open Issue?
        if max_width is None:
            max_width = get_options()["display_width"]
        srs = _format_crs(self.crs, max_width=max_width)
        return f"{self.__class__.__name__} (crs={srs})"

    def _check_crs(self, other_crs: CRS | None, allow_none: bool = False) -> bool:
        """Check if the index's projection is the same than the given one.
        If allow_none is True, empty CRS is treated as the same.
        """
        # TODO: ignore_axis_order https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.equals
        if allow_none:
            if self.crs is None or other_crs is None:
                return True
        if not self.crs == other_crs:
            return False
        return True

    def _crs_mismatch_raise(self, other_crs: CRS | None, stacklevel: int = 3):
        """Raise a CRS mismatch error or warning with the information
        on the assigned CRS.
        """
        srs = _format_crs(self.crs, max_width=50)
        other_srs = _format_crs(other_crs, max_width=50)

        msg = (
            "CRS mismatch between the CRS of index geometries "
            "and the CRS of input geometries.\n"
            f"Index CRS: {srs}\n"
            f"Input CRS: {other_srs}\n"
        )

        if get_option(CRS_INDEX_ERROR):
            raise ValueError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=stacklevel)

    def equals(self, other: Index) -> bool:
        if not isinstance(other, CRSIndex):
            return False
        if not self._check_crs(other.crs, allow_none=True):
            return False

    def join(self, other, how="inner"):
        if not self._check_crs(other.crs, allow_none=True):
            self._crs_mismatch_raise(other.crs)

        new_indexes = {
            k: v.join(other._indexes[k], how=how) for k, v in self._indexes.items()
        }

        return type(self)(new_indexes, crs=self.crs)

    def reindex_like(self, other, method=None, tolerance=None) -> dict[Hashable, Any]:
        if not self._check_crs(other.crs, allow_none=True):
            self._crs_mismatch_raise(other.crs)

        new_index = {
            k: get_indexer_nd(
                self._indexes[k].index, other._indexes[k].index, method, tolerance
            )
            for k in self._indexes.keys()
        }

        return new_index
