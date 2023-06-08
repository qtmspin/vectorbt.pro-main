import os
from datetime import datetime

import pytest
from numba import njit

import vectorbtpro as vbt
from vectorbtpro.base import (
    wrapping,
    grouping,
    combining,
    indexes,
    indexing,
    flex_indexing,
    reshaping,
    resampling,
    merging,
)
from vectorbtpro.utils import checks

from tests.utils import *

day_dt = np.timedelta64(86400000000000)

# Initialize global variables
a1 = np.array([1])
a2 = np.array([1, 2, 3])
a3 = np.array([[1, 2, 3]])
a4 = np.array([[1], [2], [3]])
a5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
sr_none = pd.Series([1])
sr1 = pd.Series([1], index=pd.Index(["x1"], name="i1"), name="a1")
sr2 = pd.Series([1, 2, 3], index=pd.Index(["x2", "y2", "z2"], name="i2"), name="a2")
df_none = pd.DataFrame([[1]])
df1 = pd.DataFrame([[1]], index=pd.Index(["x3"], name="i3"), columns=pd.Index(["a3"], name="c3"))
df2 = pd.DataFrame([[1], [2], [3]], index=pd.Index(["x4", "y4", "z4"], name="i4"), columns=pd.Index(["a4"], name="c4"))
df3 = pd.DataFrame([[1, 2, 3]], index=pd.Index(["x5"], name="i5"), columns=pd.Index(["a5", "b5", "c5"], name="c5"))
df4 = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    index=pd.Index(["x6", "y6", "z6"], name="i6"),
    columns=pd.Index(["a6", "b6", "c6"], name="c6"),
)
multi_i = pd.MultiIndex.from_arrays([["x7", "y7", "z7"], ["x8", "y8", "z8"]], names=["i7", "i8"])
multi_c = pd.MultiIndex.from_arrays([["a7", "b7", "c7"], ["a8", "b8", "c8"]], names=["c7", "c8"])
df5 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=multi_i, columns=multi_c)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.broadcasting["index_from"] = "stack"
    vbt.settings.broadcasting["columns_from"] = "stack"


def teardown_module():
    vbt.settings.reset()


# ############# grouping ############# #


grouped_index = pd.MultiIndex.from_arrays(
    [[1, 1, 1, 1, 0, 0, 0, 0], [3, 3, 2, 2, 1, 1, 0, 0], [7, 6, 5, 4, 3, 2, 1, 0]],
    names=["first", "second", "third"],
)


class TestGrouper:
    def test_group_by_to_index(self):
        assert not grouping.base.group_by_to_index(grouped_index, group_by=False)
        assert grouping.base.group_by_to_index(grouped_index, group_by=None) is None
        assert_index_equal(
            grouping.base.group_by_to_index(grouped_index, group_by=True),
            pd.Index(["group"] * len(grouped_index), name="group"),
        )
        assert_index_equal(
            grouping.base.group_by_to_index(grouped_index, group_by=0),
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            grouping.base.group_by_to_index(grouped_index, group_by="first"),
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            grouping.base.group_by_to_index(grouped_index, group_by=[0, 1]),
            pd.MultiIndex.from_tuples(
                [(1, 3), (1, 3), (1, 2), (1, 2), (0, 1), (0, 1), (0, 0), (0, 0)],
                names=["first", "second"],
            ),
        )
        assert_index_equal(
            grouping.base.group_by_to_index(grouped_index, group_by=["first", "second"]),
            pd.MultiIndex.from_tuples(
                [(1, 3), (1, 3), (1, 2), (1, 2), (0, 1), (0, 1), (0, 0), (0, 0)],
                names=["first", "second"],
            ),
        )
        assert_index_equal(
            grouping.base.group_by_to_index(grouped_index, group_by=np.array([3, 2, 1, 1, 1, 0, 0, 0])),
            pd.Index([3, 2, 1, 1, 1, 0, 0, 0], dtype="int64", name="group"),
        )
        assert_index_equal(
            grouping.base.group_by_to_index(
                grouped_index,
                group_by=pd.Index([3, 2, 1, 1, 1, 0, 0, 0], name="fourth"),
            ),
            pd.Index([3, 2, 1, 1, 1, 0, 0, 0], dtype="int64", name="fourth"),
        )

    def test_get_groups_and_index(self):
        a, b = grouping.base.get_groups_and_index(grouped_index, group_by=None)
        np.testing.assert_array_equal(a, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        assert_index_equal(b, grouped_index)
        a, b = grouping.base.get_groups_and_index(grouped_index, group_by=0)
        np.testing.assert_array_equal(a, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert_index_equal(b, pd.Index([1, 0], dtype="int64", name="first"))
        a, b = grouping.base.get_groups_and_index(grouped_index, group_by=[0, 1])
        np.testing.assert_array_equal(a, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
        assert_index_equal(
            b,
            pd.MultiIndex.from_tuples([(1, 3), (1, 2), (0, 1), (0, 0)], names=["first", "second"]),
        )

    def test_get_group_lens_nb(self):
        np.testing.assert_array_equal(
            grouping.nb.get_group_lens_nb(np.array([0, 0, 0, 0, 1, 1, 1, 1])),
            np.array([4, 4]),
        )
        np.testing.assert_array_equal(grouping.nb.get_group_lens_nb(np.array([0, 1])), np.array([1, 1]))
        np.testing.assert_array_equal(grouping.nb.get_group_lens_nb(np.array([0, 0])), np.array([2]))
        np.testing.assert_array_equal(grouping.nb.get_group_lens_nb(np.array([0])), np.array([1]))
        np.testing.assert_array_equal(grouping.nb.get_group_lens_nb(np.array([])), np.array([]))
        with pytest.raises(Exception):
            grouping.nb.get_group_lens_nb(np.array([1, 1, 0, 0]))
        with pytest.raises(Exception):
            grouping.nb.get_group_lens_nb(np.array([0, 1, 0, 1]))

    def test_get_group_map_nb(self):
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0, 1, 0, 1, 0, 1, 0, 1]), 2)[0],
            np.array([0, 2, 4, 6, 1, 3, 5, 7]),
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0, 1, 0, 1, 0, 1, 0, 1]), 2)[1],
            np.array([4, 4]),
        )
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([1, 0]), 2)[0], np.array([1, 0]))
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([1, 0]), 2)[1], np.array([1, 1]))
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([0, 0]), 1)[0], np.array([0, 1]))
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([0, 0]), 1)[1], np.array([2]))
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([0]), 1)[0], np.array([0]))
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([0]), 1)[1], np.array([1]))
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([], dtype=np.int_), 0)[0], np.array([]))
        np.testing.assert_array_equal(grouping.nb.get_group_map_nb(np.array([], dtype=np.int_), 0)[1], np.array([]))

    def test_is_grouped(self):
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped()
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped(group_by=1)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouped(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouped()
        assert vbt.Grouper(grouped_index).is_grouped(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouped(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouped(group_by=False)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_enabled(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled()
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(group_by=True)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(group_by=1)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_enabled()
        assert vbt.Grouper(grouped_index).is_grouping_enabled(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouping_enabled(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_enabled(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_disabled(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled()
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(group_by=True)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_disabled()
        assert not vbt.Grouper(grouped_index).is_grouping_disabled(group_by=0)
        assert not vbt.Grouper(grouped_index).is_grouping_disabled(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_disabled(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_modified(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_modified()
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_modified()
        assert vbt.Grouper(grouped_index).is_grouping_modified(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouping_modified(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_modified(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_changed(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_changed()
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_changed()
        assert vbt.Grouper(grouped_index).is_grouping_changed(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouping_changed(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_changed(group_by=False)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_group_count_changed(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_group_count_changed()
        assert vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(group_by=False)
        assert not vbt.Grouper(grouped_index).is_group_count_changed()
        assert vbt.Grouper(grouped_index).is_group_count_changed(group_by=0)
        assert vbt.Grouper(grouped_index).is_group_count_changed(group_by=True)
        assert not vbt.Grouper(grouped_index).is_group_count_changed(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_check_group_by(self):
        vbt.Grouper(grouped_index, group_by=None, allow_enable=True).check_group_by(group_by=0)
        with pytest.raises(Exception):
            vbt.Grouper(grouped_index, group_by=None, allow_enable=False).check_group_by(group_by=0)
        vbt.Grouper(grouped_index, group_by=0, allow_disable=True).check_group_by(group_by=False)
        with pytest.raises(Exception):
            vbt.Grouper(grouped_index, group_by=0, allow_disable=False).check_group_by(group_by=False)
        vbt.Grouper(grouped_index, group_by=0, allow_modify=True).check_group_by(group_by=1)
        vbt.Grouper(grouped_index, group_by=0, allow_modify=False).check_group_by(
            group_by=np.array([2, 2, 2, 2, 3, 3, 3, 3]),
        )
        with pytest.raises(Exception):
            vbt.Grouper(grouped_index, group_by=0, allow_modify=False).check_group_by(group_by=1)

    def test_resolve_group_by(self):
        assert vbt.Grouper(grouped_index, group_by=None).resolve_group_by() is None  # default
        assert_index_equal(
            vbt.Grouper(grouped_index, group_by=None).resolve_group_by(group_by=0),  # overrides
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            vbt.Grouper(grouped_index, group_by=0).resolve_group_by(),  # default
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            vbt.Grouper(grouped_index, group_by=0).resolve_group_by(group_by=1),  # overrides
            pd.Index([3, 3, 2, 2, 1, 1, 0, 0], dtype="int64", name="second"),
        )

    def test_get_groups(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_groups(),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_groups(group_by=0),
            np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        )

    def test_get_index(self):
        assert_index_equal(
            vbt.Grouper(grouped_index).get_index(),
            vbt.Grouper(grouped_index).index,
        )
        assert_index_equal(
            vbt.Grouper(grouped_index).get_index(group_by=0),
            pd.Index([1, 0], dtype="int64", name="first"),
        )

    def test_get_group_lens(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_lens(),
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_lens(group_by=0),
            np.array([4, 4]),
        )

    def test_get_group_start_idxs(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_start_idxs(),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_start_idxs(group_by=0),
            np.array([0, 4]),
        )

    def test_get_group_end_idxs(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_end_idxs(),
            np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_end_idxs(group_by=0),
            np.array([4, 8]),
        )

    def test_yield_group_idxs(self):
        np.testing.assert_array_equal(
            np.concatenate(tuple(vbt.Grouper(grouped_index).yield_group_idxs())),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )
        np.testing.assert_array_equal(
            np.concatenate(tuple(vbt.Grouper(grouped_index).yield_group_idxs(group_by=0))),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )

    def test_eq(self):
        assert vbt.Grouper(grouped_index) == vbt.Grouper(grouped_index)
        assert vbt.Grouper(grouped_index, group_by=0) == vbt.Grouper(grouped_index, group_by=0)
        assert vbt.Grouper(grouped_index) != 0
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, group_by=0)
        assert vbt.Grouper(grouped_index) != vbt.Grouper(pd.Index([0]))
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, allow_enable=False)
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, allow_disable=False)
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, allow_modify=False)


# ############# resampling ############# #


class TestResampler:
    @pytest.mark.parametrize("test_freq", ["1h", "3d", "7d"])
    @pytest.mark.parametrize("test_inclusive", ["left", "right"])
    def test_date_range_nb(self, test_freq, test_inclusive):
        source_index = pd.date_range("2020-01-01", "2020-02-01")
        np.testing.assert_array_equal(
            resampling.nb.date_range_nb(
                source_index[0].to_datetime64(),
                source_index[-1].to_datetime64(),
                pd.Timedelta(test_freq).to_timedelta64(),
                incl_left=test_inclusive == "left",
                incl_right=test_inclusive == "right",
            ),
            pd.date_range(source_index[0], source_index[-1], freq=test_freq, inclusive=test_inclusive).values,
        )

    def test_from_pd_resample(self):
        source_index = pd.date_range("2020-01-01", "2020-02-01", freq="1h")
        resampler = vbt.Resampler.from_pd_resample(source_index, "1d")
        target_index = pd.Series(index=source_index).resample("1d").count().index
        assert_index_equal(resampler.source_index, source_index)
        assert_index_equal(resampler.target_index, target_index)
        assert resampler.source_freq == source_index.freq
        assert resampler.target_freq == target_index.freq

    def test_from_pd_date_range(self):
        source_index = pd.date_range("2020-01-01", "2020-02-01", freq="1h")
        resampler = vbt.Resampler.from_pd_date_range(source_index, "2020-01-01", "2020-02-01", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="1d")
        assert_index_equal(resampler.source_index, source_index)
        assert_index_equal(resampler.target_index, target_index)
        assert resampler.source_freq == source_index.freq
        assert resampler.target_freq == target_index.freq

    def test_downsample_map_to_target_index(self):
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True),
            pd.DatetimeIndex(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-15",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-06", "2020-02-01", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False),
            np.array([-1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False),
            pd.DatetimeIndex(
                [
                    pd.NaT,
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-13",
                    "2020-01-13",
                    "2020-01-13",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, before=True),
            np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-01-14", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False, before=True)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False, before=True),
            np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )

    def test_upsample_map_to_target_index(self):
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False),
            np.array([9, 12, 14, 16, 19, 21, 24, 26, 28, 31, 33]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True),
            pd.DatetimeIndex(
                [
                    "2020-01-04 18:00:00",
                    "2020-01-06 00:00:00",
                    "2020-01-06 20:00:00",
                    "2020-01-07 16:00:00",
                    "2020-01-08 22:00:00",
                    "2020-01-09 18:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-11 20:00:00",
                    "2020-01-12 16:00:00",
                    "2020-01-13 22:00:00",
                    "2020-01-14 18:00:00",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-06", "2020-02-01", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False),
            np.array([-1, 0, 2, 4, 7, 9, 12, 14, 16, 19, 21]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False),
            pd.DatetimeIndex(
                [
                    pd.NaT,
                    "2020-01-06 00:00:00",
                    "2020-01-06 20:00:00",
                    "2020-01-07 16:00:00",
                    "2020-01-08 22:00:00",
                    "2020-01-09 18:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-11 20:00:00",
                    "2020-01-12 16:00:00",
                    "2020-01-13 22:00:00",
                    "2020-01-14 18:00:00",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, before=True),
            np.array([10, 12, 15, 17, 20, 22, 24, 27, 29, 32, 34]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-05 04:00:00",
                    "2020-01-06 00:00:00",
                    "2020-01-07 06:00:00",
                    "2020-01-08 02:00:00",
                    "2020-01-09 08:00:00",
                    "2020-01-10 04:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-13 02:00:00",
                    "2020-01-14 08:00:00",
                    "2020-01-15 04:00:00",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-01-14", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False, before=True)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False, before=True),
            np.array([10, 12, 15, 17, 20, 22, 24, 27, 29, -1, -1]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-05 04:00:00",
                    "2020-01-06 00:00:00",
                    "2020-01-07 06:00:00",
                    "2020-01-08 02:00:00",
                    "2020-01-09 08:00:00",
                    "2020-01-10 04:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-13 02:00:00",
                    pd.NaT,
                    pd.NaT,
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )

    def test_index_difference(self):
        source_index = ["2020-01-01", "2020-01-02", "2020-01-03"]
        target_index = ["2020-01-01T12:00:00", "2020-01-02T00:00:00", "2020-01-03T00:00:00", "2020-01-03T12:00:00"]
        resampler = vbt.Resampler(source_index, target_index)
        assert_index_equal(
            resampler.index_difference(), pd.DatetimeIndex(["2020-01-01 12:00:00"], dtype="datetime64[ns]", freq=None)
        )
        assert_index_equal(
            resampler.index_difference(reverse=True),
            pd.DatetimeIndex(["2020-01-01 12:00:00", "2020-01-03 12:00:00"], dtype="datetime64[ns]", freq=None),
        )


# ############# wrapping ############# #


sr2_wrapper = vbt.ArrayWrapper.from_obj(sr2)
df2_wrapper = vbt.ArrayWrapper.from_obj(df2)
df4_wrapper = vbt.ArrayWrapper.from_obj(df4)

sr2_wrapper_co = sr2_wrapper.replace(column_only_select=True)
df4_wrapper_co = df4_wrapper.replace(column_only_select=True)

sr2_grouped_wrapper = sr2_wrapper.replace(group_by=np.array(["g1"]), group_select=True)
df4_grouped_wrapper = df4_wrapper.replace(group_by=np.array(["g1", "g1", "g2"]), group_select=True)

sr2_grouped_wrapper_co = sr2_grouped_wrapper.replace(column_only_select=True, group_select=True)
df4_grouped_wrapper_co = df4_grouped_wrapper.replace(column_only_select=True, group_select=True)

sr2_grouped_wrapper_conog = sr2_grouped_wrapper.replace(column_only_select=True, group_select=False)
df4_grouped_wrapper_conog = df4_grouped_wrapper.replace(column_only_select=True, group_select=False)


class TestArrayWrapper:
    def test_row_stack(self):
        assert vbt.ArrayWrapper.row_stack(
            (
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
            )
        ) == vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.RangeIndex(start=0, stop=3), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.RangeIndex(start=3, stop=6), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.RangeIndex(start=0, stop=6, step=1))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.RangeIndex(start=0, stop=3, name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.RangeIndex(start=3, stop=6, name="i"), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.RangeIndex(start=0, stop=6, step=1, name="i"))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.RangeIndex(start=0, stop=3, name="i1"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.RangeIndex(start=3, stop=6, name="i2"), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.RangeIndex(start=0, stop=3), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.RangeIndex(start=4, stop=7), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 4, 5, 6], dtype="int64"))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 4, 5, 6]))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([2, 3, 4]), pd.Index(["a", "b"], name="c"), 2),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([2, 3, 4]), pd.Index(["a", "b"], name="c"), 2),
            verify_integrity=False,
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 2, 3, 4]))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([2, 1, 0]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([3, 4, 5]), pd.Index(["a", "b"], name="c"), 2),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([2, 1, 0]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([3, 4, 5]), pd.Index(["a", "b"], name="c"), 2),
            verify_integrity=False,
        )
        assert_index_equal(wrapper.index, pd.Index([2, 1, 0, 3, 4, 5]))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index(["x", "y", "z"]), pd.Index(["a", "b"], name="c"), 2),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index(["x", "y", "z"]), pd.Index(["a", "b"], name="c"), 2),
            verify_integrity=False,
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, "x", "y", "z"]))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index(["x", "y", "z"]), pd.Index(["a", "b"], name="c"), 2),
            index=np.arange(6),
        )
        assert_index_equal(wrapper.index, pd.Index(np.arange(6)))
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=1)
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-06", inclusive="left")
        index3 = pd.date_range("2020-01-07", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2.append(index3), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2).append(index3))
        assert wrapper.freq == pd.Timedelta(days=1)
        with pytest.raises(Exception):
            index1 = pd.date_range("2020-01-01", "2020-01-05", freq="1d", inclusive="left")
            index2 = pd.date_range("2020-01-05", "2020-01-10", freq="1h", inclusive="left")
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
            )
        index1 = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
        index2 = pd.DatetimeIndex(["2020-01-06", "2020-01-10"])
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq is None
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
            freq="3d",
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=3)
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]), pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["c1", "c2"]), 2
            ),
        )
        assert_index_equal(wrapper.columns, pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["c1", "c2"]))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]),
                pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["c1", "c2"]),
                2,
            ),
            index_stack_kwargs=dict(drop_duplicates=False),
        )
        assert_index_equal(
            wrapper.columns, pd.MultiIndex.from_tuples([("a", "a", "a"), ("b", "b", "b")], names=["c1", "c1", "c2"])
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c2"), 2),
            columns=["a2", "b2"],
        )
        assert_index_equal(wrapper.columns, pd.Index(["a2", "b2"]))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a2"], name="c2"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 4, 5, 6]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples([("a", "a2"), ("b", "a2")], names=["c1", "c2"]),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=False),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=False),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index(["a", "b"], name="c", dtype="object"),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=True),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=True),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index(["group", "group"], dtype="object", name="group"),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=[0, 1]),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=[0, 1]),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1], dtype="int64", name="group"),
        )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=True),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=False),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=[0, 1]),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=[1, 0]),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(
                pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, some_arg=2, check_expected_keys_=False
            ),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, some_arg=2, check_expected_keys_=False
            ),
        )
        assert wrapper.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, some_arg=2, check_expected_keys_=False
                ),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, some_arg=3, check_expected_keys_=False
                ),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, some_arg=2, check_expected_keys_=False
                ),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, some_arg=2, check_expected_keys_=False
                ),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, column_only_select=True),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, column_only_select=False),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, allow_enable=True),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, allow_enable=False),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, allow_enable=True),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, allow_enable=False),
            allow_enable=False,
        )
        assert not wrapper.grouper.allow_enable
        columns = pd.Index(["a", "b"], name="c")
        grouper = vbt.Grouper(columns, group_by=True)
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), columns, 2, grouper=grouper),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), columns, 2, grouper=grouper),
        )
        assert wrapper.grouper == grouper
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), columns, 2, group_by=True, allow_enable=True),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), columns, 2, group_by=False, allow_enable=False),
            grouper=grouper,
        )
        assert wrapper.grouper == grouper
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), columns, 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), columns, 2),
            grouper=grouper,
            group_by=False,
        )
        assert not wrapper.grouper.is_grouped()

    def test_column_stack(self):
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2], name="i"))
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([1, 2, 3], name="i"), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 3], name="i"))
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([2, 1, 0], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2], name="i"))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([0, 0, 1], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["c", "d"], name="c"), 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([2, 1, 0], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([2, 1, 0], name="i"), pd.Index(["c", "d"], name="c"), 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index(["x", "y", "z"], name="i"), pd.Index(["c", "d"], name="c"), 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([1, 2, 3], name="i"), pd.Index(["c", "d"], name="c"), 2),
                union_index=False,
            )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([5, 6, 7], name="i"), pd.Index(["c", "d"], name="c"), 2),
            index=[0, 1, 2, 3, 4, 5],
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 3, 4, 5]))
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=1)
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-06", inclusive="left")
        index3 = pd.date_range("2020-01-07", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2.append(index3), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2).append(index3))
        assert wrapper.freq == pd.Timedelta(days=1)
        with pytest.raises(Exception):
            index1 = pd.date_range("2020-01-01", "2020-01-05", freq="1d", inclusive="left")
            index2 = pd.date_range("2020-01-05", "2020-01-10", freq="1h", inclusive="left")
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
            )
        index1 = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
        index2 = pd.DatetimeIndex(["2020-01-06", "2020-01-10"])
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq is None
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
            freq="3d",
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=3)
        index = pd.Index([0, 1, 2], name="i")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c", "d"], name="c"))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
                vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
                normalize_columns=False,
            )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c", "d"]))
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c"), 2),
            keys=["o1", "o2"],
        )
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples([("o1", "a"), ("o1", "b"), ("o2", "c"), ("o2", "d")], names=[None, "c"]),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
            keys=pd.Index(["k1", "k2"], name="k"),
        )
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples([("k1", "a"), ("k1", "b"), ("k2", "c"), ("k2", "d")], names=["k", None]),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
            columns=["a2", "b2", "c2", "d2"],
        )
        assert_index_equal(wrapper.columns, pd.Index(["a2", "b2", "c2", "d2"]))
        columns1 = pd.Index(["a", "b"], name="c")
        columns2 = pd.Index(["c", "d"], name="c")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([2, 3], name="g")),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, 2, 3], name="g", dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=False),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, "c", "d"], dtype="object"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([0, 1], name="g")),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, 2, 3], name="group_idx", dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g1")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([2, 3], name="g2")),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, 2, 3], dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=True),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 0, 1, 1], name="group_idx", dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=True),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples(
                [("o1", "group"), ("o1", "group"), ("o2", "group"), ("o2", "group")], names=(None, "group")
            ),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=False),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples([("o1", "group"), ("o1", "group"), ("o2", "c"), ("o2", "d")], names=(None, None)),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=False),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples([("o1", "group"), ("o1", "group"), ("o2", "c"), ("o2", "d")], names=(None, None)),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([2, 3], name="g")),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples([("o1", 0), ("o1", 1), ("o2", 2), ("o2", 3)], names=(None, "g")),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, column_only_select=False),
            vbt.ArrayWrapper(index, columns2, 2, column_only_select=False),
        )
        assert not wrapper.column_only_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, column_only_select=True),
            vbt.ArrayWrapper(index, columns2, 2, column_only_select=False),
        )
        assert wrapper.column_only_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_select=True),
            vbt.ArrayWrapper(index, columns2, 2, group_select=True),
        )
        assert wrapper.group_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_select=True),
            vbt.ArrayWrapper(index, columns2, 2, group_select=False),
        )
        assert not wrapper.group_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_enable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_enable=True),
        )
        assert wrapper.grouper.allow_enable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_enable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_enable=False),
        )
        assert not wrapper.grouper.allow_enable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_disable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_disable=True),
        )
        assert wrapper.grouper.allow_disable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_disable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_disable=False),
        )
        assert not wrapper.grouper.allow_disable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_modify=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_modify=True),
        )
        assert wrapper.grouper.allow_modify
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_modify=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_modify=False),
        )
        assert not wrapper.grouper.allow_modify
        columns = pd.Index(["a", "b", "c", "d"], name="c2")
        grouper = vbt.Grouper(columns, group_by=True)
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2),
                vbt.ArrayWrapper(index, columns2, 2),
                grouper=grouper,
            )
        columns = pd.Index(["a", "b", "c", "d"], name="c")
        grouper = vbt.Grouper(columns, group_by=True)
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_modify=False),
            vbt.ArrayWrapper(index, columns2, 2, allow_modify=False),
            grouper=grouper,
        )
        assert wrapper.grouper == grouper
        assert wrapper.grouper.allow_modify
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, some_arg=2, check_expected_keys_=False),
            vbt.ArrayWrapper(index, columns2, 2, some_arg=2, check_expected_keys_=False),
        )
        assert wrapper.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2, some_arg=2, check_expected_keys_=False),
                vbt.ArrayWrapper(index, columns2, 2, some_arg=3, check_expected_keys_=False),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2, some_arg=2, check_expected_keys_=False),
                vbt.ArrayWrapper(index, columns2, 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2),
                vbt.ArrayWrapper(index, columns2, 2, some_arg=2, check_expected_keys_=False),
            )

    def test_config(self, tmp_path):
        assert vbt.ArrayWrapper.loads(sr2_wrapper.dumps()) == sr2_wrapper
        assert vbt.ArrayWrapper.loads(sr2_wrapper_co.dumps()) == sr2_wrapper_co
        assert vbt.ArrayWrapper.loads(sr2_grouped_wrapper.dumps()) == sr2_grouped_wrapper
        assert vbt.ArrayWrapper.loads(sr2_grouped_wrapper_co.dumps()) == sr2_grouped_wrapper_co
        sr2_grouped_wrapper_co.save(tmp_path / "sr2_grouped_wrapper_co")
        assert vbt.ArrayWrapper.load(tmp_path / "sr2_grouped_wrapper_co") == sr2_grouped_wrapper_co
        sr2_grouped_wrapper_co.save(tmp_path / "sr2_grouped_wrapper_co", file_format="ini")
        assert vbt.ArrayWrapper.load(tmp_path / "sr2_grouped_wrapper_co", file_format="ini") == sr2_grouped_wrapper_co

    def test_indexing_func_meta(self):
        # not grouped
        wrapper_meta = sr2_wrapper.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[0, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 1, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 2, None)
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[:2, 0])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[:2, [0]])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(0, 1, None)
        assert wrapper_meta["group_idxs"] == slice(0, 1, None)
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[:2, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 2, None)
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 2], [0, 2]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 2]))
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 2]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 2]))
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 0], [0, 0]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[0, 0])
        assert wrapper_meta["row_idxs"] == slice(0, 1, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0

        # not grouped, column only
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[0])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 1, None)
        assert wrapper_meta["group_idxs"] == slice(0, 1, None)
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 2, None)
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 2]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 2]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 2]))
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        with pytest.raises(Exception):
            sr2_wrapper_co.indexing_func_meta(lambda x: x.iloc[:2])
        with pytest.raises(Exception):
            df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[:, :2])

        # grouped
        wrapper_meta = sr2_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, 0])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, 1])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 2
        assert wrapper_meta["group_idxs"] == 1
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, [1]])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(2, 3, None)
        assert wrapper_meta["group_idxs"] == slice(1, 2, None)
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 2], [0, 1]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 2]))
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 0], [0, 0]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[0, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 1, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)

        # grouped, column only
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[0])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[1])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 2
        assert wrapper_meta["group_idxs"] == 1
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[[1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(2, 3, None)
        assert wrapper_meta["group_idxs"] == slice(1, 2, None)
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))

        # grouped, column only, no group select
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[0])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[1])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 1
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[[1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(1, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 1, None)
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[[0, 1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[[0, 0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))

    def test_indexing(self):
        # not grouped
        assert_index_equal(sr2_wrapper.iloc[:2].index, pd.Index(["x2", "y2"], dtype="object", name="i2"))
        assert_index_equal(sr2_wrapper.iloc[:2].columns, pd.Index(["a2"], dtype="object"))
        assert sr2_wrapper.iloc[:2].ndim == 1
        assert_index_equal(df4_wrapper.iloc[0, :2].index, pd.Index(["x6"], dtype="object", name="i6"))
        assert_index_equal(df4_wrapper.iloc[0, :2].columns, pd.Index(["a6", "b6"], dtype="object", name="c6"))
        assert df4_wrapper.iloc[0, :2].ndim == 2
        assert_index_equal(df4_wrapper.iloc[:2, 0].index, pd.Index(["x6", "y6"], dtype="object", name="i6"))
        assert_index_equal(df4_wrapper.iloc[:2, 0].columns, pd.Index(["a6"], dtype="object", name="c6"))
        assert df4_wrapper.iloc[:2, 0].ndim == 1
        assert_index_equal(
            df4_wrapper.iloc[:2, [0]].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(df4_wrapper.iloc[:2, [0]].columns, pd.Index(["a6"], dtype="object", name="c6"))
        assert df4_wrapper.iloc[:2, [0]].ndim == 2
        assert_index_equal(df4_wrapper.iloc[:2, :2].index, pd.Index(["x6", "y6"], dtype="object", name="i6"))
        assert_index_equal(
            df4_wrapper.iloc[:2, :2].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_wrapper.iloc[:2, :2].ndim == 2

        # not grouped, column only
        assert_index_equal(
            df4_wrapper_co.iloc[0].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(df4_wrapper_co.iloc[0].columns, pd.Index(["a6"], dtype="object", name="c6"))
        assert df4_wrapper_co.iloc[0].ndim == 1
        assert_index_equal(
            df4_wrapper_co.iloc[[0]].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(df4_wrapper_co.iloc[[0]].columns, pd.Index(["a6"], dtype="object", name="c6"))
        assert df4_wrapper_co.iloc[[0]].ndim == 2
        assert_index_equal(
            df4_wrapper_co.iloc[:2].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_wrapper_co.iloc[:2].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_wrapper_co.iloc[:2].ndim == 2

        # grouped
        assert_index_equal(
            sr2_grouped_wrapper.iloc[:2].index,
            pd.Index(["x2", "y2"], dtype="object", name="i2"),
        )
        assert_index_equal(sr2_grouped_wrapper.iloc[:2].columns, pd.Index(["a2"], dtype="object"))
        assert sr2_grouped_wrapper.iloc[:2].ndim == 1
        assert sr2_grouped_wrapper.iloc[:2].grouped_ndim == 1
        assert_index_equal(
            sr2_grouped_wrapper.iloc[:2].grouper.group_by, pd.Index(["g1"], dtype="object", name="group")
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 0].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 0].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, 0].ndim == 2
        assert df4_grouped_wrapper.iloc[:2, 0].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 0].grouper.group_by,
            pd.Index(["g1", "g1"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 1].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 1].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, 1].ndim == 1
        assert df4_grouped_wrapper.iloc[:2, 1].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 1].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, [1]].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, [1]].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, [1]].ndim == 2
        assert df4_grouped_wrapper.iloc[:2, [1]].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, [1]].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, :2].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, :2].columns,
            pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, :2].ndim == 2
        assert df4_grouped_wrapper.iloc[:2, :2].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, :2].grouper.group_by,
            pd.Index(["g1", "g1", "g2"], dtype="object", name="group"),
        )

        # grouped, column only
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[0].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[0].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[0].ndim == 2
        assert df4_grouped_wrapper_co.iloc[0].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[0].grouper.group_by,
            pd.Index(["g1", "g1"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[1].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[1].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[1].ndim == 1
        assert df4_grouped_wrapper_co.iloc[1].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[1].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[[1]].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[[1]].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[[1]].ndim == 2
        assert df4_grouped_wrapper_co.iloc[[1]].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[[1]].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[:2].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[:2].columns,
            pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[:2].ndim == 2
        assert df4_grouped_wrapper_co.iloc[:2].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[:2].grouper.group_by,
            pd.Index(["g1", "g1", "g2"], dtype="object", name="group"),
        )

    def test_from_obj(self):
        assert vbt.ArrayWrapper.from_obj(sr2) == sr2_wrapper
        assert vbt.ArrayWrapper.from_obj(df4) == df4_wrapper
        assert vbt.ArrayWrapper.from_obj(sr2, column_only_select=True) == sr2_wrapper_co
        assert vbt.ArrayWrapper.from_obj(df4, column_only_select=True) == df4_wrapper_co

    def test_from_shape(self):
        assert vbt.ArrayWrapper.from_shape((3,)) == vbt.ArrayWrapper(
            pd.RangeIndex(start=0, stop=3, step=1),
            pd.RangeIndex(start=0, stop=1, step=1),
            1,
        )
        assert vbt.ArrayWrapper.from_shape((3, 3)) == vbt.ArrayWrapper.from_obj(
            pd.DataFrame(np.empty((3, 3))),
        )

    def test_columns(self):
        assert_index_equal(df4_wrapper.columns, df4.columns)
        assert_index_equal(df4_grouped_wrapper.columns, df4.columns)
        assert_index_equal(df4_grouped_wrapper.get_columns(), pd.Index(["g1", "g2"], dtype="object", name="group"))

    def test_name(self):
        assert sr2_wrapper.name == "a2"
        assert df4_wrapper.name is None
        assert vbt.ArrayWrapper.from_obj(pd.Series([0])).name is None
        assert sr2_grouped_wrapper.name == "a2"
        assert sr2_grouped_wrapper.get_name() == "g1"
        assert df4_grouped_wrapper.name is None
        assert df4_grouped_wrapper.get_name() is None

    def test_ndim(self):
        assert sr2_wrapper.ndim == 1
        assert df4_wrapper.ndim == 2
        assert sr2_grouped_wrapper.ndim == 1
        assert sr2_grouped_wrapper.get_ndim() == 1
        assert df4_grouped_wrapper.ndim == 2
        assert df4_grouped_wrapper.get_ndim() == 2
        assert df4_grouped_wrapper["g1"].ndim == 2
        assert df4_grouped_wrapper["g1"].get_ndim() == 1
        assert df4_grouped_wrapper["g2"].ndim == 1
        assert df4_grouped_wrapper["g2"].get_ndim() == 1

    def test_shape(self):
        assert sr2_wrapper.shape == (3,)
        assert df4_wrapper.shape == (3, 3)
        assert sr2_grouped_wrapper.shape == (3,)
        assert sr2_grouped_wrapper.get_shape() == (3,)
        assert df4_grouped_wrapper.shape == (3, 3)
        assert df4_grouped_wrapper.get_shape() == (3, 2)

    def test_shape_2d(self):
        assert sr2_wrapper.shape_2d == (3, 1)
        assert df4_wrapper.shape_2d == (3, 3)
        assert sr2_grouped_wrapper.shape_2d == (3, 1)
        assert sr2_grouped_wrapper.get_shape_2d() == (3, 1)
        assert df4_grouped_wrapper.shape_2d == (3, 3)
        assert df4_grouped_wrapper.get_shape_2d() == (3, 2)

    def test_freq(self):
        assert sr2_wrapper.freq is None
        assert sr2_wrapper.replace(freq="1D").freq == day_dt
        assert (
            sr2_wrapper.replace(index=pd.Index([datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)])).freq
            == day_dt
        )

    def test_period(self):
        test_sr = pd.Series([1, 2], index=[datetime(2020, 1, 1), datetime(2021, 1, 1)])
        assert test_sr.vbt.wrapper.period == 2

    def test_dt_period(self):
        assert sr2_wrapper.dt_period == 3
        assert sr2_wrapper.replace(freq="1D").dt_period == 3
        test_sr = pd.Series([1, 2], index=[datetime(2020, 1, 1), datetime(2021, 1, 1)])
        assert test_sr.vbt.wrapper.dt_period == 2
        assert test_sr.vbt(freq="1D").wrapper.dt_period == 367

    def test_to_timedelta(self):
        sr = pd.Series([1, 2, np.nan], index=["x", "y", "z"], name="name")
        assert_series_equal(
            vbt.ArrayWrapper.from_obj(sr, freq="1 days").arr_to_timedelta(sr),
            pd.Series(
                np.array([86400000000000, 172800000000000, "NaT"], dtype="timedelta64[ns]"),
                index=sr.index,
                name=sr.name,
            ),
        )
        df = sr.to_frame()
        assert_frame_equal(
            vbt.ArrayWrapper.from_obj(df, freq="1 days").arr_to_timedelta(df),
            pd.DataFrame(
                np.array([86400000000000, 172800000000000, "NaT"], dtype="timedelta64[ns]"),
                index=df.index,
                columns=df.columns,
            ),
        )

    def test_wrap(self):
        assert_series_equal(
            vbt.ArrayWrapper(index=sr1.index, columns=[0], ndim=1).wrap(a1),  # empty
            pd.Series(a1, index=sr1.index, name=None),
        )
        assert_series_equal(
            vbt.ArrayWrapper(index=sr1.index, columns=[sr1.name], ndim=1).wrap(a1),
            pd.Series(a1, index=sr1.index, name=sr1.name),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=sr1.index, columns=[sr1.name], ndim=2).wrap(a1),
            pd.DataFrame(a1, index=sr1.index, columns=[sr1.name]),
        )
        assert_series_equal(
            vbt.ArrayWrapper(index=sr2.index, columns=[sr2.name], ndim=1).wrap(a2),
            pd.Series(a2, index=sr2.index, name=sr2.name),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=sr2.index, columns=[sr2.name], ndim=2).wrap(a2),
            pd.DataFrame(a2, index=sr2.index, columns=[sr2.name]),
        )
        assert_series_equal(
            vbt.ArrayWrapper(index=df2.index, columns=df2.columns, ndim=1).wrap(a2),
            pd.Series(a2, index=df2.index, name=df2.columns[0]),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df2.index, columns=df2.columns, ndim=2).wrap(a2),
            pd.DataFrame(a2, index=df2.index, columns=df2.columns),
        )
        assert_frame_equal(
            vbt.ArrayWrapper.from_obj(df2).wrap(a2, index=df4.index),
            pd.DataFrame(a2, index=df4.index, columns=df2.columns),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df4.index, columns=df4.columns, ndim=2).wrap(
                np.array([[0, 0, np.nan], [1, np.nan, 1], [2, 2, np.nan]]),
                fillna=-1,
            ),
            pd.DataFrame([[0.0, 0.0, -1.0], [1.0, -1.0, 1.0], [2.0, 2.0, -1.0]], index=df4.index, columns=df4.columns),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df4.index, columns=df4.columns, ndim=2).wrap(
                np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                to_index=True,
            ),
            pd.DataFrame(
                [["x6", "x6", "x6"], ["y6", "y6", "y6"], ["z6", "z6", "z6"]],
                index=df4.index,
                columns=df4.columns,
            ),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df4.index, columns=df4.columns, ndim=2, freq="d").wrap(
                np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                to_timedelta=True,
            ),
            pd.DataFrame(
                [
                    [pd.Timedelta(days=0), pd.Timedelta(days=0), pd.Timedelta(days=0)],
                    [pd.Timedelta(days=1), pd.Timedelta(days=1), pd.Timedelta(days=1)],
                    [pd.Timedelta(days=2), pd.Timedelta(days=2), pd.Timedelta(days=2)],
                ],
                index=df4.index,
                columns=df4.columns,
            ),
        )

    def test_wrap_reduced(self):
        # sr to value
        assert sr2_wrapper.wrap_reduced(0) == 0
        assert sr2_wrapper.wrap_reduced(np.array([0])) == 0  # result of computation on 2d
        # sr to array
        assert_series_equal(
            sr2_wrapper.wrap_reduced(np.array([0, 1])),
            pd.Series(np.array([0, 1]), name=sr2.name),
        )
        assert_series_equal(
            sr2_wrapper.wrap_reduced(np.array([0, 1]), name_or_index=["x", "y"]),
            pd.Series(np.array([0, 1]), index=["x", "y"], name=sr2.name),
        )
        assert_series_equal(
            sr2_wrapper.wrap_reduced(np.array([0, 1]), name_or_index=["x", "y"], columns=[0]),
            pd.Series(np.array([0, 1]), index=["x", "y"], name=None),
        )
        # df to value
        assert df2_wrapper.wrap_reduced(0) == 0
        assert df4_wrapper.wrap_reduced(0) == 0
        # df to value per column
        assert_series_equal(
            df4_wrapper.wrap_reduced(np.array([0, 1, 2]), name_or_index="test"),
            pd.Series(np.array([0, 1, 2]), index=df4.columns, name="test"),
        )
        assert_series_equal(
            df4_wrapper.wrap_reduced(np.array([0, 1, 2]), columns=["m", "n", "l"], name_or_index="test"),
            pd.Series(np.array([0, 1, 2]), index=["m", "n", "l"], name="test"),
        )
        # df to array per column
        assert_frame_equal(
            df4_wrapper.wrap_reduced(np.array([[0, 1, 2], [3, 4, 5]]), name_or_index=["x", "y"]),
            pd.DataFrame(np.array([[0, 1, 2], [3, 4, 5]]), index=["x", "y"], columns=df4.columns),
        )
        assert_frame_equal(
            df4_wrapper.wrap_reduced(
                np.array([[0, 1, 2], [3, 4, 5]]),
                name_or_index=["x", "y"],
                columns=["m", "n", "l"],
            ),
            pd.DataFrame(np.array([[0, 1, 2], [3, 4, 5]]), index=["x", "y"], columns=["m", "n", "l"]),
        )

    def test_grouped_wrapping(self):
        assert_frame_equal(
            df4_grouped_wrapper_co.wrap(np.array([[1, 2], [3, 4], [5, 6]])),
            pd.DataFrame(
                np.array([[1, 2], [3, 4], [5, 6]]),
                index=df4.index,
                columns=pd.Index(["g1", "g2"], dtype="object", name="group"),
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.wrap_reduced(np.array([1, 2])),
            pd.Series(np.array([1, 2]), index=pd.Index(["g1", "g2"], dtype="object", name="group")),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.wrap(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), group_by=False),
            pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index=df4.index, columns=df4.columns),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.wrap_reduced(np.array([1, 2, 3]), group_by=False),
            pd.Series(np.array([1, 2, 3]), index=df4.columns),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[0].wrap(np.array([1, 2, 3])),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name="g1"),
        )
        assert df4_grouped_wrapper_co.iloc[0].wrap_reduced(np.array([1])) == 1
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[0].wrap(np.array([[1], [2], [3]])),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name="g1"),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[0].wrap(np.array([[1, 2], [3, 4], [5, 6]]), group_by=False),
            pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), index=df4.index, columns=df4.columns[:2]),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[0].wrap_reduced(np.array([1, 2]), group_by=False),
            pd.Series(np.array([1, 2]), index=df4.columns[:2]),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap(np.array([1, 2, 3])),
            pd.DataFrame(
                np.array([[1], [2], [3]]),
                index=df4.index,
                columns=pd.Index(["g1"], dtype="object", name="group"),
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap_reduced(np.array([1])),
            pd.Series(np.array([1]), index=pd.Index(["g1"], dtype="object", name="group")),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap(np.array([[1, 2], [3, 4], [5, 6]]), group_by=False),
            pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), index=df4.index, columns=df4.columns[:2]),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap_reduced(np.array([1, 2]), group_by=False),
            pd.Series(np.array([1, 2]), index=df4.columns[:2]),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[1].wrap(np.array([1, 2, 3])),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name="g2"),
        )
        assert df4_grouped_wrapper_co.iloc[1].wrap_reduced(np.array([1])) == 1
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[1].wrap(np.array([1, 2, 3]), group_by=False),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name=df4.columns[2]),
        )
        assert df4_grouped_wrapper_co.iloc[1].wrap_reduced(np.array([1]), group_by=False) == 1
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap(np.array([1, 2, 3])),
            pd.DataFrame(
                np.array([[1], [2], [3]]),
                index=df4.index,
                columns=pd.Index(["g2"], dtype="object", name="group"),
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap_reduced(np.array([1])),
            pd.Series(np.array([1]), index=pd.Index(["g2"], dtype="object", name="group")),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap(np.array([1, 2, 3]), group_by=False),
            pd.DataFrame(np.array([[1], [2], [3]]), index=df4.index, columns=df4.columns[2:]),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap_reduced(np.array([1]), group_by=False),
            pd.Series(np.array([1]), index=df4.columns[2:]),
        )

    def test_dummy(self):
        assert_index_equal(sr2_wrapper.dummy().index, sr2_wrapper.index)
        assert_index_equal(sr2_wrapper.dummy().to_frame().columns, sr2_wrapper.columns)
        assert_index_equal(df4_wrapper.dummy().index, df4_wrapper.index)
        assert_index_equal(df4_wrapper.dummy().columns, df4_wrapper.columns)
        assert_index_equal(sr2_grouped_wrapper.dummy().index, sr2_grouped_wrapper.index)
        assert_index_equal(
            sr2_grouped_wrapper.dummy().to_frame().columns.rename("group"),
            sr2_grouped_wrapper.get_columns(),
        )
        assert_index_equal(df4_grouped_wrapper.dummy().index, df4_grouped_wrapper.index)
        assert_index_equal(df4_grouped_wrapper.dummy().columns, df4_grouped_wrapper.get_columns())

    def test_fill(self):
        assert_series_equal(sr2_wrapper.fill(0), sr2 * 0)
        assert_frame_equal(df4_wrapper.fill(0), df4 * 0)
        assert_series_equal(
            sr2_grouped_wrapper.fill(0),
            pd.Series(0, index=sr2.index, name="g1"),
        )
        assert_frame_equal(
            df4_grouped_wrapper.fill(0),
            pd.DataFrame(0, index=df4.index, columns=pd.Index(["g1", "g2"], name="group")),
        )

    def test_fill_reduced(self):
        assert sr2_wrapper.fill_reduced(0) == 0
        assert_series_equal(df4_wrapper.fill_reduced(0), pd.Series(0, index=df4.columns))
        assert sr2_grouped_wrapper.fill_reduced(0) == 0
        assert_series_equal(
            df4_grouped_wrapper.fill_reduced(0),
            pd.Series(0, index=pd.Index(["g1", "g2"], name="group")),
        )

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        ts = pd.Series(np.arange(5), index=pd.date_range("2020-01-01", "2020-01-05"))
        assert_index_equal(
            ts.vbt.wrapper.resample(test_freq).index,
            ts.resample(test_freq).last().index,
        )
        assert ts.vbt.wrapper.resample(test_freq).freq == ts.resample(test_freq).last().vbt.wrapper.freq

    def test_fill_and_set(self):
        i = pd.date_range("2020-01-01", "2020-01-05")
        c = pd.Index(["a", "b", "c"], name="c")
        sr = pd.Series(np.nan, index=i, name=c[0])
        sr_wrapper = wrapping.ArrayWrapper.from_obj(sr)
        df = pd.DataFrame(np.nan, index=i, columns=c)
        df_wrapper = wrapping.ArrayWrapper.from_obj(df)

        obj = sr_wrapper.fill_and_set(
            indexing.index_dict(
                {
                    vbt.rowidx(0): 100,
                    "_def": 0,
                }
            )
        )
        assert_series_equal(
            obj,
            pd.Series([100, 0, 0, 0, 0], index=i, name=c[0]),
        )
        obj = df_wrapper.fill_and_set(
            indexing.index_dict(
                {
                    vbt.idx(1, 1): 100,
                    "_def": 1,
                }
            )
        )
        assert_frame_equal(
            obj,
            pd.DataFrame(
                [
                    [1, 1, 1],
                    [1, 100, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                index=i,
                columns=c,
            ),
        )

        def _sr_assert_flex_index_dct(index_dct, target_arr):
            arr = sr_wrapper.fill_and_set(indexing.index_dict(index_dct), keep_flex=True)
            np.testing.assert_array_equal(arr, target_arr)

        _sr_assert_flex_index_dct({indexing.hslice(None, None, None): 0}, np.array([0.0]))
        _sr_assert_flex_index_dct({0: 0}, np.array([0.0, np.nan, np.nan, np.nan, np.nan]))
        _sr_assert_flex_index_dct({(0, 2): 0}, np.array([0.0, np.nan, 0.0, np.nan, np.nan]))
        _sr_assert_flex_index_dct({(0, 2): [0, 1]}, np.array([0.0, np.nan, 1.0, np.nan, np.nan]))
        _sr_assert_flex_index_dct(
            {(0, 2): vbt.RepEval("np.arange(len(row_idxs))")}, np.array([0.0, np.nan, 1.0, np.nan, np.nan])
        )
        _sr_assert_flex_index_dct(
            {indexing.idx(0, indexing.hslice(None, None, None)): 0}, np.array([0, np.nan, np.nan, np.nan, np.nan])
        )

        def _df_assert_flex_index_dct(index_dct, target_arr):
            arr = df_wrapper.fill_and_set(indexing.index_dict(index_dct), keep_flex=True)
            np.testing.assert_array_equal(arr, target_arr)

        _df_assert_flex_index_dct({indexing.rowidx(indexing.hslice(None, None, None)): 0}, np.array([0.0])[:, None])
        _df_assert_flex_index_dct({indexing.rowidx(0): 0}, np.array([0.0, np.nan, np.nan, np.nan, np.nan])[:, None])
        _df_assert_flex_index_dct({indexing.rowidx((0, 2)): 0}, np.array([0.0, np.nan, 0.0, np.nan, np.nan])[:, None])
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): [0, 1]}, np.array([0.0, np.nan, 1.0, np.nan, np.nan])[:, None]
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): vbt.RepEval("np.arange(len(row_idxs))")},
            np.array([0.0, np.nan, 1.0, np.nan, np.nan])[:, None],
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx(0): [0, 1, 2]},
            np.array(
                [
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): [[0, 1, 2]]},
            np.array(
                [
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): [[0, 1, 2], [0, 1, 2]]},
            np.array(
                [
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct({indexing.rowidx((0, 2)): 0}, np.array([0.0, np.nan, 0.0, np.nan, np.nan])[:, None])
        _df_assert_flex_index_dct({indexing.colidx(indexing.hslice(None, None, None)): 0}, np.array([0.0])[None])
        _df_assert_flex_index_dct({indexing.colidx(0): 0}, np.array([0.0, np.nan, np.nan])[None])
        _df_assert_flex_index_dct({indexing.colidx((0, 2)): 0}, np.array([0.0, np.nan, 0.0])[None])
        _df_assert_flex_index_dct({indexing.colidx((0, 2)): [0, 1]}, np.array([0.0, np.nan, 1.0])[None])
        _df_assert_flex_index_dct(
            {indexing.colidx((0, 2)): vbt.RepEval("np.arange(len(col_idxs))")}, np.array([0.0, np.nan, 1.0])[None]
        )
        _df_assert_flex_index_dct(
            {indexing.colidx(0): [0, 1, 2, 3, 4]},
            np.array(
                [
                    [0, np.nan, np.nan],
                    [1, np.nan, np.nan],
                    [2, np.nan, np.nan],
                    [3, np.nan, np.nan],
                    [4, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.colidx((0, 2)): [[0], [1], [2], [3], [4]]},
            np.array(
                [
                    [0, np.nan, 0],
                    [1, np.nan, 1],
                    [2, np.nan, 2],
                    [3, np.nan, 3],
                    [4, np.nan, 4],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.colidx((0, 2)): [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]},
            np.array(
                [
                    [0, np.nan, 0],
                    [1, np.nan, 1],
                    [2, np.nan, 2],
                    [3, np.nan, 3],
                    [4, np.nan, 4],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx(indexing.hslice(None, None, None), indexing.hslice(None, None, None)): 0},
            np.array([0.0])[None],
        )
        _df_assert_flex_index_dct(
            {indexing.idx(0, 0): 0},
            np.array(
                [
                    [0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), 0): 0},
            np.array(
                [
                    [0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx(0, (0, 2)): 0},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): 0},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [0, 1]},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [1, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [[0], [1]]},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [1, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [[0, 1]]},
            np.array(
                [
                    [0, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [0, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [[0, 1], [2, 3]]},
            np.array(
                [
                    [0, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [2, np.nan, 3],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )


sr2_wrapping = wrapping.Wrapping(sr2_wrapper)
df4_wrapping = wrapping.Wrapping(df4_wrapper)

sr2_grouped_wrapping = wrapping.Wrapping(sr2_grouped_wrapper)
df4_grouped_wrapping = wrapping.Wrapping(df4_grouped_wrapper)


class TestWrapping:
    def test_regroup(self):
        assert df4_wrapping.regroup(None) == df4_wrapping
        assert df4_wrapping.regroup(False) == df4_wrapping
        assert df4_grouped_wrapping.regroup(None) == df4_grouped_wrapping
        assert df4_grouped_wrapping.regroup(df4_grouped_wrapper.grouper.group_by) == df4_grouped_wrapping
        assert_index_equal(
            df4_wrapping.regroup(df4_grouped_wrapper.grouper.group_by).wrapper.grouper.group_by,
            df4_grouped_wrapper.grouper.group_by,
        )
        assert df4_grouped_wrapping.regroup(False).wrapper.grouper.group_by is None

    def test_select_col(self):
        assert sr2_wrapping.select_col() == sr2_wrapping
        assert sr2_grouped_wrapping.select_col() == sr2_grouped_wrapping
        assert_index_equal(
            df4_wrapping.select_col(column="a6").wrapper.get_columns(),
            pd.Index(["a6"], dtype="object", name="c6"),
        )
        assert_index_equal(
            df4_grouped_wrapping.select_col(column="g1").wrapper.get_columns(),
            pd.Index(["g1"], dtype="object", name="group"),
        )
        with pytest.raises(Exception):
            df4_wrapping.select_col()
        with pytest.raises(Exception):
            df4_grouped_wrapping.select_col()


# ############# indexes ############# #


class TestIndexes:
    def test_get_index(self):
        assert_index_equal(indexes.get_index(sr1, 0), sr1.index)
        assert_index_equal(indexes.get_index(sr1, 1), pd.Index([sr1.name]))
        assert_index_equal(indexes.get_index(pd.Series([1, 2, 3]), 1), pd.Index([0]))  # empty
        assert_index_equal(indexes.get_index(df1, 0), df1.index)
        assert_index_equal(indexes.get_index(df1, 1), df1.columns)

    def test_index_from_values(self):
        assert_index_equal(
            indexes.index_from_values([0.1, 0.2], name="a"),
            pd.Index([0.1, 0.2], dtype="float64", name="a"),
        )
        assert_index_equal(
            indexes.index_from_values(np.tile(np.arange(1, 4)[:, None][:, None], (1, 3, 3)), name="b"),
            pd.Index([1, 2, 3], dtype="int64", name="b"),
        )
        assert_index_equal(
            indexes.index_from_values(
                [
                    np.random.uniform(size=(3, 3)),
                    np.random.uniform(size=(3, 3)),
                    np.random.uniform(size=(3, 3)),
                ],
                name="c",
            ),
            pd.Index(["array_0", "array_1", "array_2"], dtype="object", name="c"),
        )
        rand_arr = np.random.uniform(size=(3, 3))
        assert_index_equal(
            indexes.index_from_values([rand_arr, rand_arr, rand_arr], name="c"),
            pd.Index(["array_0", "array_0", "array_0"], dtype="object", name="c"),
        )
        assert_index_equal(
            indexes.index_from_values(
                [
                    rand_arr,
                    np.random.uniform(size=(3, 3)),
                    rand_arr,
                    np.random.uniform(size=(3, 3)),
                ],
                name="c",
            ),
            pd.Index(["array_0", "array_1", "array_0", "array_2"], dtype="object", name="c"),
        )
        assert_index_equal(
            indexes.index_from_values([(1, 2), (3, 4), (5, 6)], name="c"),
            pd.Index(["tuple_0", "tuple_1", "tuple_2"], dtype="object", name="c"),
        )

        class A:
            pass

        class B:
            pass

        class C:
            pass

        assert_index_equal(
            indexes.index_from_values([A(), B(), B(), C()], name="c"),
            pd.Index(["A_0", "B_0", "B_1", "C_0"], dtype="object", name="c"),
        )
        a = A()
        b = B()
        c = C()
        assert_index_equal(
            indexes.index_from_values([a, b, b, c], name="c"),
            pd.Index(["A_0", "B_0", "B_0", "C_0"], dtype="object", name="c"),
        )

    def test_repeat_index(self):
        i = pd.Index([1, 2, 3], name="i")
        assert_index_equal(
            indexes.repeat_index(i, 3),
            pd.Index([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype="int64", name="i"),
        )
        assert_index_equal(
            indexes.repeat_index(multi_i, 3),
            pd.MultiIndex.from_tuples(
                [
                    ("x7", "x8"),
                    ("x7", "x8"),
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("y7", "y8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                    ("z7", "z8"),
                    ("z7", "z8"),
                ],
                names=["i7", "i8"],
            ),
        )
        assert_index_equal(indexes.repeat_index([0], 3), pd.Index([0, 1, 2], dtype="int64"))  # empty
        assert_index_equal(
            indexes.repeat_index(sr_none.index, 3), pd.RangeIndex(start=0, stop=3, step=1)  # simple range,
        )

    def test_tile_index(self):
        i = pd.Index([1, 2, 3], name="i")
        assert_index_equal(
            indexes.tile_index(i, 3),
            pd.Index([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype="int64", name="i"),
        )
        assert_index_equal(
            indexes.tile_index(multi_i, 3),
            pd.MultiIndex.from_tuples(
                [
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                ],
                names=["i7", "i8"],
            ),
        )
        assert_index_equal(indexes.tile_index([0], 3), pd.Index([0, 1, 2], dtype="int64"))  # empty
        assert_index_equal(
            indexes.tile_index(sr_none.index, 3), pd.RangeIndex(start=0, stop=3, step=1)  # simple range,
        )

    def test_stack_indexes(self):
        assert_index_equal(
            indexes.stack_indexes([sr2.index, df2.index, df5.index]),
            pd.MultiIndex.from_tuples(
                [("x2", "x4", "x7", "x8"), ("y2", "y4", "y7", "y8"), ("z2", "z4", "z7", "z8")],
                names=["i2", "i4", "i7", "i8"],
            ),
        )
        assert_index_equal(
            indexes.stack_indexes([sr2.index, df2.index, sr2.index], drop_duplicates=False),
            pd.MultiIndex.from_tuples(
                [("x2", "x4", "x2"), ("y2", "y4", "y2"), ("z2", "z4", "z2")],
                names=["i2", "i4", "i2"],
            ),
        )
        assert_index_equal(
            indexes.stack_indexes([sr2.index, df2.index, sr2.index], drop_duplicates=True),
            pd.MultiIndex.from_tuples([("x4", "x2"), ("y4", "y2"), ("z4", "z2")], names=["i4", "i2"]),
        )
        assert_index_equal(
            indexes.stack_indexes([pd.Index([1, 1]), pd.Index([2, 3])], drop_redundant=True),
            pd.Index([2, 3]),
        )

    def test_combine_indexes(self):
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1]), pd.Index([2, 3])], drop_redundant=False),
            pd.MultiIndex.from_tuples([(1, 2), (1, 3)]),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1]), pd.Index([2, 3])], drop_redundant=True),
            pd.Index([2, 3], dtype="int64"),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1], name="i"), pd.Index([2, 3])], drop_redundant=True),
            pd.MultiIndex.from_tuples([(1, 2), (1, 3)], names=["i", None]),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1, 2]), pd.Index([3])], drop_redundant=False),
            pd.MultiIndex.from_tuples([(1, 3), (2, 3)]),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1, 2]), pd.Index([3])], drop_redundant=True),
            pd.Index([1, 2], dtype="int64"),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1]), pd.Index([2, 3])], drop_redundant=(False, True)),
            pd.Index([2, 3], dtype="int64"),
        )
        assert_index_equal(
            indexes.combine_indexes([df2.index, df5.index]),
            pd.MultiIndex.from_tuples(
                [
                    ("x4", "x7", "x8"),
                    ("x4", "y7", "y8"),
                    ("x4", "z7", "z8"),
                    ("y4", "x7", "x8"),
                    ("y4", "y7", "y8"),
                    ("y4", "z7", "z8"),
                    ("z4", "x7", "x8"),
                    ("z4", "y7", "y8"),
                    ("z4", "z7", "z8"),
                ],
                names=["i4", "i7", "i8"],
            ),
        )

    def test_drop_levels(self):
        assert_index_equal(
            indexes.drop_levels(multi_i, "i7"),
            pd.Index(["x8", "y8", "z8"], dtype="object", name="i8"),
        )
        assert_index_equal(
            indexes.drop_levels(multi_i, "i8"),
            pd.Index(["x7", "y7", "z7"], dtype="object", name="i7"),
        )
        assert_index_equal(indexes.drop_levels(multi_i, "i9", strict=False), multi_i)
        with pytest.raises(Exception):
            indexes.drop_levels(multi_i, "i9")
        assert_index_equal(
            indexes.drop_levels(multi_i, ["i7", "i8"], strict=False),  # won't do anything
            pd.MultiIndex.from_tuples([("x7", "x8"), ("y7", "y8"), ("z7", "z8")], names=["i7", "i8"]),
        )
        with pytest.raises(Exception):
            indexes.drop_levels(multi_i, ["i7", "i8"])

    def test_rename_levels(self):
        i = pd.Index([1, 2, 3], name="i")
        assert_index_equal(
            indexes.rename_levels(i, {"i": "f"}),
            pd.Index([1, 2, 3], dtype="int64", name="f"),
        )
        assert_index_equal(indexes.rename_levels(i, {"a": "b"}, strict=False), i)
        with pytest.raises(Exception):
            indexes.rename_levels(i, {"a": "b"}, strict=True)
        assert_index_equal(
            indexes.rename_levels(multi_i, {"i7": "f7", "i8": "f8"}),
            pd.MultiIndex.from_tuples([("x7", "x8"), ("y7", "y8"), ("z7", "z8")], names=["f7", "f8"]),
        )

    def test_select_levels(self):
        assert_index_equal(
            indexes.select_levels(multi_i, "i7"),
            pd.Index(["x7", "y7", "z7"], dtype="object", name="i7"),
        )
        assert_index_equal(
            indexes.select_levels(multi_i, ["i7"]),
            pd.MultiIndex.from_tuples([("x7",), ("y7",), ("z7",)], names=["i7"]),
        )
        assert_index_equal(
            indexes.select_levels(multi_i, ["i7", "i8"]),
            pd.MultiIndex.from_tuples([("x7", "x8"), ("y7", "y8"), ("z7", "z8")], names=["i7", "i8"]),
        )

    def test_drop_redundant_levels(self):
        assert_index_equal(
            indexes.drop_redundant_levels(pd.Index(["a", "a"])),
            pd.Index(["a", "a"], dtype="object"),
        )  # if one unnamed, leaves as-is
        assert_index_equal(
            indexes.drop_redundant_levels(pd.MultiIndex.from_arrays([["a", "a"], ["b", "b"]])),
            pd.MultiIndex.from_tuples([("a", "b"), ("a", "b")]),  # if all unnamed, leaves as-is
        )
        assert_index_equal(
            indexes.drop_redundant_levels(pd.MultiIndex.from_arrays([["a", "a"], ["b", "b"]], names=["hi", None])),
            pd.Index(["a", "a"], dtype="object", name="hi"),  # removes level with single unnamed value
        )
        assert_index_equal(
            indexes.drop_redundant_levels(pd.MultiIndex.from_arrays([["a", "b"], ["a", "b"]], names=["hi", "hi2"])),
            pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["hi", "hi2"]),  # legit
        )
        assert_index_equal(  # ignores 0-to-n
            indexes.drop_redundant_levels(pd.MultiIndex.from_arrays([[0, 1], ["a", "b"]], names=[None, "hi2"])),
            pd.Index(["a", "b"], dtype="object", name="hi2"),
        )
        assert_index_equal(  # legit
            indexes.drop_redundant_levels(pd.MultiIndex.from_arrays([[0, 2], ["a", "b"]], names=[None, "hi2"])),
            pd.MultiIndex.from_tuples([(0, "a"), (2, "b")], names=[None, "hi2"]),
        )
        assert_index_equal(  # legit (w/ name)
            indexes.drop_redundant_levels(pd.MultiIndex.from_arrays([[0, 1], ["a", "b"]], names=["hi", "hi2"])),
            pd.MultiIndex.from_tuples([(0, "a"), (1, "b")], names=["hi", "hi2"]),
        )

    def test_drop_duplicate_levels(self):
        assert_index_equal(
            indexes.drop_duplicate_levels(pd.MultiIndex.from_arrays([[1, 2, 3], [1, 2, 3]], names=["a", "a"])),
            pd.Index([1, 2, 3], dtype="int64", name="a"),
        )
        assert_index_equal(
            indexes.drop_duplicate_levels(
                pd.MultiIndex.from_tuples([(0, 1, 2, 1), ("a", "b", "c", "b")], names=["x", "y", "z", "y"]),
                keep="last",
            ),
            pd.MultiIndex.from_tuples([(0, 2, 1), ("a", "c", "b")], names=["x", "z", "y"]),
        )
        assert_index_equal(
            indexes.drop_duplicate_levels(
                pd.MultiIndex.from_tuples([(0, 1, 2, 1), ("a", "b", "c", "b")], names=["x", "y", "z", "y"]),
                keep="first",
            ),
            pd.MultiIndex.from_tuples([(0, 1, 2), ("a", "b", "c")], names=["x", "y", "z"]),
        )

    def test_align_index_to(self):
        index1 = pd.Index(["c", "b", "a"], name="name1")
        assert indexes.align_index_to(index1, index1) == pd.IndexSlice[:]
        index2 = pd.Index(["a", "b", "c", "a", "b", "c"], name="name1")
        np.testing.assert_array_equal(indexes.align_index_to(index1, index2), np.array([2, 1, 0, 2, 1, 0]))
        with pytest.raises(Exception):
            indexes.align_index_to(pd.Index(["a"]), pd.Index(["a", "b", "c"]))
        index3 = pd.MultiIndex.from_tuples(
            [(0, "c"), (0, "b"), (0, "a"), (1, "c"), (1, "b"), (1, "a")],
            names=["name2", "name1"],
        )
        np.testing.assert_array_equal(indexes.align_index_to(index1, index3), np.array([0, 1, 2, 0, 1, 2]))
        with pytest.raises(Exception):
            indexes.align_index_to(pd.Index(["b", "a"], name="name1"), index3)
        with pytest.raises(Exception):
            indexes.align_index_to(pd.Index(["c", "b", "a", "a"], name="name1"), index3)
        index4 = pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
            names=["name2", "name1"],
        )
        np.testing.assert_array_equal(indexes.align_index_to(index1, index4), np.array([2, 1, 0, 2, 1, 0]))

    def test_align_indexes(self):
        index1 = pd.Index(["a", "b", "c"])
        index2 = pd.MultiIndex.from_tuples([(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")])
        index3 = pd.MultiIndex.from_tuples(
            [
                (2, 0, "a"),
                (2, 0, "b"),
                (2, 0, "c"),
                (2, 1, "a"),
                (2, 1, "b"),
                (2, 1, "c"),
                (3, 0, "a"),
                (3, 0, "b"),
                (3, 0, "c"),
                (3, 1, "a"),
                (3, 1, "b"),
                (3, 1, "c"),
            ]
        )
        indices1, indices2, indices3 = indexes.align_indexes([index1, index2, index3])
        np.testing.assert_array_equal(indices1, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(indices2, np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]))
        assert indices3 == pd.IndexSlice[:]

    def test_pick_levels(self):
        index = indexes.stack_indexes([multi_i, multi_c])
        assert indexes.pick_levels(index, required_levels=[], optional_levels=[]) == ([], [])
        assert indexes.pick_levels(index, required_levels=["c8", "c7", "i8", "i7"], optional_levels=[]) == (
            [3, 2, 1, 0],
            [],
        )
        assert indexes.pick_levels(index, required_levels=["c8", None, "i8", "i7"], optional_levels=[]) == (
            [3, 2, 1, 0],
            [],
        )
        assert indexes.pick_levels(index, required_levels=[None, "c7", "i8", "i7"], optional_levels=[]) == (
            [3, 2, 1, 0],
            [],
        )
        assert indexes.pick_levels(index, required_levels=[None, None, None, None], optional_levels=[]) == (
            [0, 1, 2, 3],
            [],
        )
        assert indexes.pick_levels(index, required_levels=["c8", "c7", "i8"], optional_levels=["i7"]) == (
            [3, 2, 1],
            [0],
        )
        assert indexes.pick_levels(index, required_levels=["c8", None, "i8"], optional_levels=["i7"]) == (
            [3, 2, 1],
            [0],
        )
        assert indexes.pick_levels(index, required_levels=[None, "c7", "i8"], optional_levels=["i7"]) == (
            [3, 2, 1],
            [0],
        )
        assert indexes.pick_levels(index, required_levels=[None, None, None, None], optional_levels=[None]) == (
            [0, 1, 2, 3],
            [None],
        )
        with pytest.raises(Exception):
            indexes.pick_levels(index, required_levels=["i8", "i8", "i8", "i8"], optional_levels=[])
        with pytest.raises(Exception):
            indexes.pick_levels(index, required_levels=["c8", "c7", "i8", "i7"], optional_levels=["i7"])

    def test_concat_indexes(self):
        assert_index_equal(
            indexes.concat_indexes(
                pd.RangeIndex(stop=2),
                pd.RangeIndex(stop=3),
            ),
            pd.RangeIndex(start=0, stop=5, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="append",
            ),
            pd.Index([4, 5, 6, 1, 2, 3], dtype="int64", name="name"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="union",
            ),
            pd.Index([1, 2, 3, 4, 5, 6], dtype="int64", name="name"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="pd_concat",
            ),
            pd.Index([4, 5, 6, 1, 2, 3], dtype="int64", name="name"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="reset",
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize",
                verify_integrity=False,
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 0], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize_each",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="append",
            ),
            pd.Index(["a", "b", "c", 1, 2, 3]),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="union",
            ),
            pd.Index(["a", "b", "c", 1, 2, 3]),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="pd_concat",
            ),
            pd.Index(["a", "b", "c", 1, 2, 3]),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="reset",
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize_each",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="append",
            ),
            pd.Index([("a", 4), ("b", 5), ("c", 6), 1, 2, 3]),
        )
        with pytest.raises(Exception):
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="union",
            )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="pd_concat",
            ),
            pd.MultiIndex.from_tuples(
                [("a", 4), ("b", 5), ("c", 6), (None, 1), (None, 2), (None, 3)], names=("name1", "name2")
            ),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="reset",
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize_each",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method=("append", "factorize_each"),
                axis=2,
            ),
            pd.Index([("a", 4), ("b", 5), ("c", 6), 1, 2, 3], dtype="object"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method=("union", "factorize_each"),
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.RangeIndex(stop=2),
                pd.RangeIndex(stop=3),
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.MultiIndex.from_tuples([("x", 0), ("x", 1), ("y", 0), ("y", 1), ("y", 2)], names=["key", None]),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="append",
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.MultiIndex.from_tuples(
                [("x", 4), ("x", 5), ("x", 6), ("y", 1), ("y", 2), ("y", 3)], names=["key", "name"]
            ),
        )
        with pytest.raises(Exception):
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="union",
                keys=pd.Index(["x", "y"], name="key"),
            )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="pd_concat",
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.MultiIndex.from_tuples(
                [("x", 4), ("x", 5), ("x", 6), ("y", 1), ("y", 2), ("y", 3)], names=["key", "name"]
            ),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="reset",
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize",
                keys=pd.Index(["x", "y"], name="key"),
                verify_integrity=False,
                axis=2,
            ),
            pd.MultiIndex.from_tuples(
                [("x", 0), ("x", 1), ("x", 2), ("y", 3), ("y", 4), ("y", 0)], names=["key", "group_idx"]
            ),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize_each",
                keys=pd.Index(["x", "y"], name="key"),
                axis=2,
            ),
            pd.MultiIndex.from_tuples(
                [("x", 0), ("x", 1), ("x", 2), ("y", 3), ("y", 4), ("y", 5)], names=["key", "group_idx"]
            ),
        )


# ############# reshaping ############# #


class TestReshaping:
    def test_soft_to_ndim(self):
        np.testing.assert_array_equal(reshaping.soft_to_ndim(a2, 1), a2)
        assert_series_equal(reshaping.soft_to_ndim(sr2, 1), sr2)
        assert_series_equal(reshaping.soft_to_ndim(df2, 1), df2.iloc[:, 0])
        assert_frame_equal(reshaping.soft_to_ndim(df4, 1), df4)  # cannot -> do nothing
        np.testing.assert_array_equal(reshaping.soft_to_ndim(a2, 2), a2[:, None])
        assert_frame_equal(reshaping.soft_to_ndim(sr2, 2), sr2.to_frame())
        assert_frame_equal(reshaping.soft_to_ndim(df2, 2), df2)

    def test_to_1d(self):
        np.testing.assert_array_equal(reshaping.to_1d(None), np.array([None]))
        np.testing.assert_array_equal(reshaping.to_1d(0), np.array([0]))
        np.testing.assert_array_equal(reshaping.to_1d(a2), a2)
        assert_series_equal(reshaping.to_1d(sr2), sr2)
        assert_series_equal(reshaping.to_1d(df2), df2.iloc[:, 0])
        np.testing.assert_array_equal(reshaping.to_1d(df2, raw=True), df2.iloc[:, 0].values)

    def test_to_2d(self):
        np.testing.assert_array_equal(reshaping.to_2d(None), np.array([[None]]))
        np.testing.assert_array_equal(reshaping.to_2d(0), np.array([[0]]))
        np.testing.assert_array_equal(reshaping.to_2d(a2), a2[:, None])
        assert_frame_equal(reshaping.to_2d(sr2), sr2.to_frame())
        assert_frame_equal(reshaping.to_2d(df2), df2)
        np.testing.assert_array_equal(reshaping.to_2d(df2, raw=True), df2.values)

    def test_repeat_axis0(self):
        target = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(reshaping.repeat(0, 3, axis=0), np.full(3, 0))
        np.testing.assert_array_equal(reshaping.repeat(a2, 3, axis=0), target)
        assert_series_equal(
            reshaping.repeat(sr2, 3, axis=0),
            pd.Series(target, index=indexes.repeat_index(sr2.index, 3), name=sr2.name),
        )
        assert_frame_equal(
            reshaping.repeat(df2, 3, axis=0),
            pd.DataFrame(target, index=indexes.repeat_index(df2.index, 3), columns=df2.columns),
        )

    def test_repeat_axis1(self):
        target = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(reshaping.repeat(0, 3, axis=1), np.full((1, 3), 0))
        np.testing.assert_array_equal(reshaping.repeat(a2, 3, axis=1), target)
        assert_frame_equal(
            reshaping.repeat(sr2, 3, axis=1),
            pd.DataFrame(target, index=sr2.index, columns=indexes.repeat_index([sr2.name], 3)),
        )
        assert_frame_equal(
            reshaping.repeat(df2, 3, axis=1),
            pd.DataFrame(target, index=df2.index, columns=indexes.repeat_index(df2.columns, 3)),
        )

    def test_tile_axis0(self):
        target = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(reshaping.tile(0, 3, axis=0), np.full(3, 0))
        np.testing.assert_array_equal(reshaping.tile(a2, 3, axis=0), target)
        assert_series_equal(
            reshaping.tile(sr2, 3, axis=0),
            pd.Series(target, index=indexes.tile_index(sr2.index, 3), name=sr2.name),
        )
        assert_frame_equal(
            reshaping.tile(df2, 3, axis=0),
            pd.DataFrame(target, index=indexes.tile_index(df2.index, 3), columns=df2.columns),
        )

    def test_tile_axis1(self):
        target = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(reshaping.tile(0, 3, axis=1), np.full((1, 3), 0))
        np.testing.assert_array_equal(reshaping.tile(a2, 3, axis=1), target)
        assert_frame_equal(
            reshaping.tile(sr2, 3, axis=1),
            pd.DataFrame(target, index=sr2.index, columns=indexes.tile_index([sr2.name], 3)),
        )
        assert_frame_equal(
            reshaping.tile(df2, 3, axis=1),
            pd.DataFrame(target, index=df2.index, columns=indexes.tile_index(df2.columns, 3)),
        )

    def test_broadcast_numpy(self):
        # 1d
        broadcasted_arrs = list(np.broadcast_arrays(0, a1, a2))
        broadcasted = reshaping.broadcast(0, a1, a2)
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(broadcasted[i], broadcasted_arrs[i])
        # 2d
        broadcasted_arrs = list(np.broadcast_arrays(0, a1, a2[:, None], a3, a4, a5))
        broadcasted = reshaping.broadcast(0, a1, a2, a3, a4, a5)
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(broadcasted[i], broadcasted_arrs[i])

    def test_broadcast_axis(self):
        x1 = np.array([1])
        x2 = np.array([[1, 2, 3]])
        x3 = np.array([[1], [2], [3]])
        dct = dict(x1=x1, x2=x2, x3=x3)
        out_dct = reshaping.broadcast(dct, axis=0)
        np.testing.assert_array_equal(out_dct["x1"], np.array([[1], [1], [1]]))
        np.testing.assert_array_equal(out_dct["x2"], np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
        np.testing.assert_array_equal(out_dct["x3"], np.array([[1], [2], [3]]))
        out_dct = reshaping.broadcast(dct, axis=1)
        np.testing.assert_array_equal(out_dct["x1"], np.array([[1, 1, 1]]))
        np.testing.assert_array_equal(out_dct["x2"], np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(out_dct["x3"], np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))

    def test_broadcast_stack(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from="stack",
            columns_from="stack",
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples([("x1", "x2"), ("x1", "y2"), ("x1", "z2")], names=["i1", "i2"]),
                    name=None,
                ),
            )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from="stack",
            columns_from="stack",
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples(
                        [
                            ("x1", "x2", "x3", "x4", "x5", "x6"),
                            ("x1", "y2", "x3", "y4", "x5", "y6"),
                            ("x1", "z2", "x3", "z4", "x5", "z6"),
                        ],
                        names=["i1", "i2", "i3", "i4", "i5", "i6"],
                    ),
                    columns=pd.MultiIndex.from_tuples(
                        [("a3", "a4", "a5", "a6"), ("a3", "a4", "b5", "b6"), ("a3", "a4", "c5", "c6")],
                        names=["c3", "c4", "c5", "c6"],
                    ),
                ),
            )

        broadcasted = reshaping.broadcast(
            pd.DataFrame([[1, 2, 3]], columns=pd.Index(["a", "b", "c"], name="i1")),
            pd.DataFrame([[4, 5, 6]], columns=pd.Index(["a", "b", "c"], name="i2")),
            index_from="stack",
            columns_from="stack",
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        assert_frame_equal(
            broadcasted[0],
            pd.DataFrame(
                [[1, 2, 3]],
                columns=pd.MultiIndex.from_tuples([("a", "a"), ("b", "b"), ("c", "c")], names=["i1", "i2"]),
            ),
        )
        assert_frame_equal(
            broadcasted[1],
            pd.DataFrame(
                [[4, 5, 6]],
                columns=pd.MultiIndex.from_tuples([("a", "a"), ("b", "b"), ("c", "c")], names=["i1", "i2"]),
            ),
        )

    def test_broadcast_keep(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from="keep",
            columns_from="keep",
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(4):
            assert_series_equal(
                broadcasted[i],
                pd.Series(broadcasted_arrs[i], index=pd.RangeIndex(start=0, stop=3, step=1)),
            )
        assert_series_equal(
            broadcasted[4],
            pd.Series(broadcasted_arrs[4], index=pd.Index(["x1", "x1", "x1"], name="i1"), name=sr1.name),
        )
        assert_series_equal(broadcasted[5], pd.Series(broadcasted_arrs[5], index=sr2.index, name=sr2.name))
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from="keep",
            columns_from="keep",
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(7):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.RangeIndex(start=0, stop=3, step=1),
                    columns=pd.RangeIndex(start=0, stop=3, step=1),
                ),
            )
        assert_frame_equal(
            broadcasted[7],
            pd.DataFrame(
                broadcasted_arrs[7],
                index=pd.Index(["x1", "x1", "x1"], dtype="object", name="i1"),
                columns=pd.Index(["a1", "a1", "a1"], dtype="object"),
            ),
        )
        assert_frame_equal(
            broadcasted[8],
            pd.DataFrame(broadcasted_arrs[8], index=sr2.index, columns=pd.Index(["a2", "a2", "a2"], dtype="object")),
        )
        assert_frame_equal(
            broadcasted[9],
            pd.DataFrame(
                broadcasted_arrs[9],
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.RangeIndex(start=0, stop=3, step=1),
            ),
        )
        assert_frame_equal(
            broadcasted[10],
            pd.DataFrame(
                broadcasted_arrs[10],
                index=pd.Index(["x3", "x3", "x3"], dtype="object", name="i3"),
                columns=pd.Index(["a3", "a3", "a3"], dtype="object", name="c3"),
            ),
        )
        assert_frame_equal(
            broadcasted[11],
            pd.DataFrame(
                broadcasted_arrs[11],
                index=df2.index,
                columns=pd.Index(["a4", "a4", "a4"], dtype="object", name="c4"),
            ),
        )
        assert_frame_equal(
            broadcasted[12],
            pd.DataFrame(
                broadcasted_arrs[12],
                index=pd.Index(["x5", "x5", "x5"], dtype="object", name="i5"),
                columns=df3.columns,
            ),
        )
        assert_frame_equal(
            broadcasted[13],
            pd.DataFrame(broadcasted_arrs[13], index=df4.index, columns=df4.columns),
        )

    def test_broadcast_specify(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from=multi_i,
            columns_from=["name"],  # should translate to Series name
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_series_equal(broadcasted[i], pd.Series(broadcasted_arrs[i], index=multi_i, name="name"))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from=multi_i,
            columns_from=[0],  # should translate to None
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_series_equal(broadcasted[i], pd.Series(broadcasted_arrs[i], index=multi_i, name=None))
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from=multi_i,
            columns_from=multi_c,
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(broadcasted_arrs[i], index=multi_i, columns=multi_c),
            )

    def test_broadcast_idx(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from=-1,
            columns_from=-1,  # should translate to Series name
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i],
                pd.Series(broadcasted_arrs[i], index=sr2.index, name=sr2.name),
            )
        with pytest.raises(Exception):
            reshaping.broadcast(
                *to_broadcast,
                index_from=0,
                columns_from=0,
                drop_duplicates=True,
                drop_redundant=True,
                ignore_sr_names=True,
                align_index=False,
            )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from=-1,
            columns_from=-1,
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(broadcasted_arrs[i], index=df4.index, columns=df4.columns),
            )

    def test_broadcast_strict(self):
        # 1d
        to_broadcast = sr1, sr2
        with pytest.raises(Exception):
            reshaping.broadcast(
                *to_broadcast,
                index_from="strict",  # changing index not allowed
                columns_from="stack",
                drop_duplicates=True,
                drop_redundant=True,
                ignore_sr_names=True,
                align_index=False,
            )
        # 2d
        to_broadcast = df1, df2
        with pytest.raises(Exception):
            reshaping.broadcast(
                *to_broadcast,
                index_from="stack",
                columns_from="strict",  # changing columns not allowed
                drop_duplicates=True,
                drop_redundant=True,
                ignore_sr_names=True,
                align_index=False,
            )

    def test_broadcast_dirty(self):
        # 1d
        to_broadcast = sr2, 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from="stack",
            columns_from="stack",
            drop_duplicates=False,
            drop_redundant=False,
            ignore_sr_names=False,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples(
                        [("x2", "x1", "x2"), ("y2", "x1", "y2"), ("z2", "x1", "z2")],
                        names=["i2", "i1", "i2"],
                    ),
                    name=("a2", "a1", "a2"),
                ),
            )

    def test_broadcast_to_shape(self):
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = []
        for x in to_broadcast:
            if isinstance(x, pd.Series):
                x = x.to_frame()
            elif np.asarray(x).ndim == 1:
                x = x[:, None]
            broadcasted_arrs.append(np.broadcast_to(x, (3, 3)))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            to_shape=(3, 3),
            index_from="stack",
            columns_from="stack",
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples([("x1", "x2"), ("x1", "y2"), ("x1", "z2")], names=["i1", "i2"]),
                    columns=None,
                ),
            )

    @pytest.mark.parametrize(
        "test_to_pd",
        [False, [False, False, False, False, False, False]],
    )
    def test_broadcast_to_pd(self, test_to_pd):
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            to_pd=test_to_pd,  # to NumPy
            index_from="stack",
            columns_from="stack",
            drop_duplicates=True,
            drop_redundant=True,
            ignore_sr_names=True,
            align_index=False,
        )
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(broadcasted[i], broadcasted_arrs[i])

    def test_broadcast_require_kwargs(self):
        a, b = reshaping.broadcast(np.empty((1,)), np.empty((1,)))  # readonly
        assert not a.flags.writeable
        assert not b.flags.writeable
        a, b = reshaping.broadcast(
            np.empty((1,)),
            np.empty((1,)),
            require_kwargs=[{"requirements": "W"}, {}],
        )  # writeable
        assert a.flags.writeable
        assert not b.flags.writeable
        a, b = reshaping.broadcast(
            np.empty((1,)),
            np.empty((1,)),
            require_kwargs=[{"requirements": ("W", "C")}, {}],
        )  # writeable, C order
        assert a.flags.writeable  # writeable since it was copied to make C order
        assert not b.flags.writeable
        assert not np.isfortran(a)
        assert not np.isfortran(b)

    def test_broadcast_mapping(self):
        result = reshaping.broadcast(dict(zero=0, a2=a2, sr2=sr2))
        assert type(result) == dict
        assert_series_equal(result["zero"], pd.Series([0, 0, 0], name=sr2.name, index=sr2.index))
        assert_series_equal(result["a2"], pd.Series([1, 2, 3], name=sr2.name, index=sr2.index))
        assert_series_equal(result["sr2"], pd.Series([1, 2, 3], name=sr2.name, index=sr2.index))

    def test_broadcast_individual(self):
        result = reshaping.broadcast(
            dict(zero=0, a2=a2, sr2=sr2),
            keep_flex={"_def": True, "sr2": False},
            require_kwargs={"_def": dict(dtype=float), "a2": dict(dtype=int)},
        )
        np.testing.assert_array_equal(result["zero"], np.array([[0.0]]))
        np.testing.assert_array_equal(result["a2"], np.array([[1], [2], [3]]))
        assert_series_equal(result["sr2"], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index))
        result = reshaping.broadcast(
            dict(
                zero=vbt.BCO(0),
                a2=vbt.BCO(a2, min_ndim=1, require_kwargs=dict(dtype=int)),
                sr2=vbt.BCO(sr2, keep_flex=False),
            ),
            keep_flex=True,
            require_kwargs=dict(dtype=float),
        )
        np.testing.assert_array_equal(result["zero"], np.array([[0.0]]))
        np.testing.assert_array_equal(result["a2"], np.array([1, 2, 3]))
        assert_series_equal(result["sr2"], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index))
        result = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex={"_def": True, 2: False},
            min_ndim={1: 1},
            require_kwargs={"_def": dict(dtype=float), 1: dict(dtype=int)},
        )
        np.testing.assert_array_equal(result[0], np.array([[0.0]]))
        np.testing.assert_array_equal(result[1], np.array([1, 2, 3]))
        assert_series_equal(result[2], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index))
        result = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex=[True, True, False],
            min_ndim=[1, 2, 1],
            require_kwargs=[dict(dtype=float), dict(dtype=int), dict(dtype=float)],
        )
        np.testing.assert_array_equal(result[0], np.array([0.0]))
        np.testing.assert_array_equal(result[1], np.array([[1], [2], [3]]))
        assert_series_equal(result[2], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index))

    def test_broadcast_refs(self):
        result = reshaping.broadcast(
            dict(
                a=vbt.Ref("b"),
                b=vbt.Ref("c"),
                c=vbt.BCO(vbt.Ref("d"), keep_flex=True),
                d=vbt.BCO(sr2, keep_flex=False),
            )
        )
        np.testing.assert_array_equal(result["a"], sr2.values[:, None])
        np.testing.assert_array_equal(result["b"], sr2.values[:, None])
        np.testing.assert_array_equal(result["c"], sr2.values[:, None])
        assert_series_equal(result["d"], sr2)

    def test_broadcast_defaults(self):
        result = reshaping.broadcast(
            dict(
                a=vbt.Ref("b"),
                b=vbt.Ref("c"),
                c=vbt.Default(vbt.BCO(vbt.Ref("d"), keep_flex=True)),
                d=vbt.BCO(vbt.Default(sr2), keep_flex=False),
            ),
            keep_wrap_default=False,
        )
        assert not isinstance(result["a"], vbt.Default)
        assert not isinstance(result["b"], vbt.Default)
        assert not isinstance(result["c"], vbt.Default)
        assert not isinstance(result["d"], vbt.Default)

        result = reshaping.broadcast(
            dict(
                a=vbt.Ref("b"),
                b=vbt.Ref("c"),
                c=vbt.Default(vbt.BCO(vbt.Ref("d"), keep_flex=True)),
                d=vbt.BCO(vbt.Default(sr2), keep_flex=False),
            ),
            keep_wrap_default=True,
        )
        assert not isinstance(result["a"], vbt.Default)
        assert not isinstance(result["b"], vbt.Default)
        assert isinstance(result["c"], vbt.Default)
        assert isinstance(result["d"], vbt.Default)
        np.testing.assert_array_equal(result["a"], sr2.values[:, None])
        np.testing.assert_array_equal(result["b"], sr2.values[:, None])
        np.testing.assert_array_equal(result["c"].value, sr2.values[:, None])
        assert_series_equal(result["d"].value, sr2)

    def test_broadcast_none(self):
        result = reshaping.broadcast(
            dict(a=None, b=vbt.Default(None), c=vbt.BCO(vbt.Ref("d")), d=vbt.BCO(None)),
            keep_wrap_default=True,
        )
        assert result["a"] is None
        assert isinstance(result["b"], vbt.Default)
        assert result["b"].value is None
        assert result["c"] is None
        assert result["d"] is None

    def test_broadcast_product(self):
        p = pd.Index([1, 2, 3], name="p")

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_p, reshaping.broadcast_to(p.values[None], _sr2_p))

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a3_p, _a3.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_df2_p, _df2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_p, reshaping.broadcast_to(np.repeat(p.values, 3)[None], _df2_p))

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_p, p.values[None])

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_a3_p, np.tile(_a3, (1, 3)))
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_sr2_p, _df2)
        np.testing.assert_array_equal(_p, np.repeat(p.values, 3)[None])

    def test_broadcast_tile(self):
        p = pd.Index([1, 2, 3], name="p")

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p = reshaping.broadcast(
            0,
            a2,
            sr2,
            tile=len(p),
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p)))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p)))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p)))

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p = reshaping.broadcast(
            0,
            a2,
            sr2,
            tile=p,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))

        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            tile=p,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_p, reshaping.broadcast_to(np.tile(p.values, 3)[None], _sr2_p))

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            tile=p,
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a3_p, _a3.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_df2_p, _df2.vbt.tile(len(p), keys=p))

        _0_p, _a2_p, a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            tile=p,
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_a3_p, _a3.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_df2_p, _df2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_p, reshaping.broadcast_to(np.tile(np.repeat(p.values, 3), 3)[None], _df2_p))

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p = reshaping.broadcast(
            0,
            a2,
            sr2,
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_sr2_p, _sr2)

        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_p, np.tile(p.values, 3)[None])

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_a3_p, np.tile(_a3, (1, 3)))
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_sr2_p, _df2)

        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_a3_p, np.tile(np.tile(_a3, (1, 3)), (1, 3)))
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_sr2_p, _df2)
        np.testing.assert_array_equal(_p, np.tile(np.repeat(p.values, 3), 3)[None])

    def test_broadcast_level(self):
        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param([1, 2]),
                b=vbt.Param([False, True]),
                c=vbt.Param(["x", "y"]),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 2, 2, 2, 2]]))
        np.testing.assert_array_equal(result["b"], np.array([[False, False, True, True, False, False, True, True]]))
        np.testing.assert_array_equal(result["c"], np.array([["x", "y", "x", "y", "x", "y", "x", "y"]]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (1, False, "x"),
                    (1, False, "y"),
                    (1, True, "x"),
                    (1, True, "y"),
                    (2, False, "x"),
                    (2, False, "y"),
                    (2, True, "x"),
                    (2, True, "y"),
                ],
                names=["a", "b", "c"],
            ),
        )
        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param([1, 2])),
                b=vbt.BCO(vbt.Param([False, True])),
                c=vbt.BCO(vbt.Param(["x", "y"])),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 2, 2, 2, 2]]))
        np.testing.assert_array_equal(result["b"], np.array([[False, False, True, True, False, False, True, True]]))
        np.testing.assert_array_equal(result["c"], np.array([["x", "y", "x", "y", "x", "y", "x", "y"]]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (1, False, "x"),
                    (1, False, "y"),
                    (1, True, "x"),
                    (1, True, "y"),
                    (2, False, "x"),
                    (2, False, "y"),
                    (2, True, "x"),
                    (2, True, "y"),
                ],
                names=["a", "b", "c"],
            ),
        )

        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param(1)),
                b=vbt.BCO(vbt.Param([False, True])),
                c=vbt.BCO(vbt.Param(["x", "y", "z"])),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 1, 1]]))
        np.testing.assert_array_equal(result["b"], np.array([[False, False, False, True, True, True]]))
        np.testing.assert_array_equal(result["c"], np.array([["x", "y", "z", "x", "y", "z"]]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(1, False, "x"), (1, False, "y"), (1, False, "z"), (1, True, "x"), (1, True, "y"), (1, True, "z")],
                names=["a", "b", "c"],
            ),
        )

        result2, wrapper2 = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param(1, level=0)),
                b=vbt.BCO(vbt.Param([False, True], level=1)),
                c=vbt.BCO(vbt.Param(["x", "y", "z"], level=2)),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], result2["a"])
        np.testing.assert_array_equal(result["b"], result2["b"])
        np.testing.assert_array_equal(result["c"], result2["c"])
        assert_index_equal(wrapper.columns, wrapper2.columns)

        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param(1, level=0)),
                b=vbt.BCO(vbt.Param([False, True], level=1)),
                c=vbt.BCO(vbt.Param(["x", "y", "z"], level=0)),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 1, 1]]))
        np.testing.assert_array_equal(result["b"], np.array([[False, True, False, True, False, True]]))
        np.testing.assert_array_equal(result["c"], np.array([["x", "x", "y", "y", "z", "z"]]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(1, "x", False), (1, "x", True), (1, "y", False), (1, "y", True), (1, "z", False), (1, "z", True)],
                names=["a", "c", "b"],
            ),
        )

        with pytest.raises(Exception):
            reshaping.broadcast(
                dict(
                    a=vbt.BCO(vbt.Param(1)),
                    b=vbt.BCO(vbt.Param([False, True], level=0)),
                    c=vbt.BCO(vbt.Param(["x", "y", "z"], level=1)),
                )
            )
        with pytest.raises(Exception):
            reshaping.broadcast(
                dict(
                    a=vbt.BCO(vbt.Param(1, level=0)),
                    b=vbt.BCO(vbt.Param([False, True], level=1)),
                    c=vbt.BCO(vbt.Param(["x", "y", "z"])),
                )
            )
        with pytest.raises(Exception):
            reshaping.broadcast(
                dict(
                    a=vbt.BCO(vbt.Param(1, level=-1)),
                    b=vbt.BCO(vbt.Param([False, True], level=0)),
                    c=vbt.BCO(vbt.Param(["x", "y", "z"], level=1)),
                )
            )
        with pytest.raises(Exception):
            reshaping.broadcast(
                dict(
                    a=vbt.BCO(vbt.Param(1, level=0)),
                    b=vbt.BCO(vbt.Param([False, True], level=1)),
                    c=vbt.BCO(vbt.Param(["x", "y", "z"], level=3)),
                )
            )

    def test_broadcast_product_keys(self):
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(
                    pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="a3"), name="a2"),
                    keys=pd.Index(["x", "y", "z"], name="a4"),
                ),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["x", "y", "z"], dtype="object", name="a4"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(
                    pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="a3"), name="a2"),
                    keys=pd.Index(["x", "y", "z"]),
                ),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["x", "y", "z"], dtype="object", name="a2"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="a3"), name="a2")),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c"], dtype="object", name="a2"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]), name="a2")),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c"], dtype="object", name="a2"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]))),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c"], dtype="object", name="a"))
        _, wrapper = reshaping.broadcast(
            dict(a=vbt.Param(pd.Series([1, 2, 3], name="a2")), sr=pd.Series([1, 2, 3])),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index([1, 2, 3], dtype="int64", name="a2"))

    def test_broadcast_meta(self):
        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0, np.array([[0]]))
        np.testing.assert_array_equal(_a2, a2[:, None])
        np.testing.assert_array_equal(_a3, a3)
        np.testing.assert_array_equal(_sr2, sr2.values[:, None])
        np.testing.assert_array_equal(_df2, df2.values)
        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=[False, True, True, True, True],
            align_index=False,
        )
        test_shape = (3, 3)
        test_index = pd.MultiIndex.from_tuples([("x2", "x4"), ("y2", "y4"), ("z2", "z4")], names=["i2", "i4"])
        test_columns = pd.Index(["a4", "a4", "a4"], name="c4", dtype="object")
        assert_frame_equal(
            _0,
            pd.DataFrame(np.zeros(test_shape, dtype=int), index=test_index, columns=test_columns),
        )
        np.testing.assert_array_equal(_a2, a2[:, None])
        np.testing.assert_array_equal(_a3, a3)
        np.testing.assert_array_equal(_sr2, sr2.values[:, None])
        np.testing.assert_array_equal(_df2, df2.values)
        _, wrapper = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            return_wrapper=True,
            align_index=False,
        )
        assert wrapper.shape == test_shape
        assert_index_equal(wrapper.index, test_index)
        assert_index_equal(wrapper.columns, test_columns)

    def test_broadcast_align(self):
        index1 = pd.date_range("2020-01-01", periods=3)
        index2 = pd.date_range("2020-01-02", periods=3)
        index3 = pd.date_range("2020-01-03", periods=3)
        columns2 = pd.MultiIndex.from_tuples([(0, "a"), (0, "b"), (1, "a"), (1, "b")])
        columns3 = pd.MultiIndex.from_tuples(
            [
                (2, 0, "a"),
                (2, 0, "b"),
                (2, 1, "a"),
                (2, 1, "b"),
                (3, 0, "a"),
                (3, 0, "b"),
                (3, 1, "a"),
                (3, 1, "b"),
            ]
        )
        sr1 = pd.Series(np.arange(len(index1)), index=index1)
        df2 = pd.DataFrame(
            np.reshape(np.arange(len(index2) * len(columns2)), (len(index2), len(columns2))),
            index=index2,
            columns=columns2,
        )
        df3 = pd.DataFrame(
            np.reshape(np.arange(len(index3) * len(columns3)), (len(index3), len(columns3))),
            index=index3,
            columns=columns3,
        )
        _df1, _df2, _df3 = reshaping.broadcast(sr1, df2, df3, align_index=True, align_columns=True)
        assert_frame_equal(
            _df1,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.date_range("2020-01-01", periods=5),
                columns=columns3,
            ),
        )
        assert_frame_equal(
            _df2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.date_range("2020-01-01", periods=5),
                columns=columns3,
            ),
        )
        assert_frame_equal(
            _df3,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                    ]
                ),
                index=pd.date_range("2020-01-01", periods=5),
                columns=columns3,
            ),
        )
        _df12, _df22, _df32 = reshaping.broadcast(
            sr1,
            df2,
            df3,
            align_index=True,
            align_columns=True,
            reindex_kwargs=dict(fill_value=0),
        )
        assert_frame_equal(_df12, _df1.fillna(0).astype(int))
        assert_frame_equal(_df22, _df2.fillna(0).astype(int))
        assert_frame_equal(_df32, _df3.fillna(0).astype(int))

    def test_broadcast_special(self):
        i = pd.date_range("2020-01-01", "2020-01-05")
        c = pd.Index(["a", "b", "c"], name="c")
        sr = pd.Series(np.nan, index=i, name=c[0])
        df = pd.DataFrame(np.nan, index=i, columns=c)
        _, obj = reshaping.broadcast(
            sr,
            indexing.index_dict(
                {
                    vbt.rowidx(0): 100,
                    "_def": 0,
                }
            ),
        )
        assert_series_equal(
            obj,
            pd.Series([100, 0, 0, 0, 0], index=i, name=c[0]),
        )
        _, obj = reshaping.broadcast(
            df,
            indexing.index_dict(
                {
                    vbt.idx(1, 1): 100,
                    "_def": 1,
                }
            ),
        )
        assert_frame_equal(
            obj,
            pd.DataFrame(
                [
                    [1, 1, 1],
                    [1, 100, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                index=i,
                columns=c,
            ),
        )
        _, obj = reshaping.broadcast(
            df,
            indexing.index_dict(
                {
                    vbt.idx(1, 1): 100,
                    "_def": 1,
                }
            ),
            keep_flex=True,
        )
        np.testing.assert_array_equal(
            obj,
            np.array(
                [
                    [1, 1, 1],
                    [1, 100, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            ),
        )
        _, obj = reshaping.broadcast(df, vbt.RepEval("wrapper.fill(0)"))
        assert_frame_equal(
            obj,
            pd.DataFrame(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                index=i,
                columns=c,
            ),
        )

    def test_broadcast_to(self):
        np.testing.assert_array_equal(reshaping.broadcast_to(0, a5, align_index=False), np.broadcast_to(0, a5.shape))
        assert_series_equal(
            reshaping.broadcast_to(0, sr2, align_index=False),
            pd.Series(np.broadcast_to(0, sr2.shape), index=sr2.index, name=sr2.name),
        )
        assert_frame_equal(
            reshaping.broadcast_to(0, df5, align_index=False),
            pd.DataFrame(np.broadcast_to(0, df5.shape), index=df5.index, columns=df5.columns),
        )
        assert_frame_equal(
            reshaping.broadcast_to(sr2, df5, align_index=False),
            pd.DataFrame(np.broadcast_to(sr2.to_frame(), df5.shape), index=df5.index, columns=df5.columns),
        )
        assert_frame_equal(
            reshaping.broadcast_to(sr2, df5, index_from=0, columns_from=0, align_index=False),
            pd.DataFrame(
                np.broadcast_to(sr2.to_frame(), df5.shape),
                index=sr2.index,
                columns=pd.Index(["a2", "a2", "a2"], dtype="object"),
            ),
        )

    @pytest.mark.parametrize(
        "test_input",
        [0, a2, a5, sr2, df5, np.zeros((2, 2, 2))],
    )
    def test_broadcast_to_array_of(self, test_input):
        # broadcasting first element to be an array out of the second argument
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of(0.1, test_input),
            np.full((1, *np.asarray(test_input).shape), 0.1),
        )
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of([0.1], test_input),
            np.full((1, *np.asarray(test_input).shape), 0.1),
        )
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of([0.1, 0.2], test_input),
            np.concatenate(
                (np.full((1, *np.asarray(test_input).shape), 0.1), np.full((1, *np.asarray(test_input).shape), 0.2)),
            ),
        )
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of(np.expand_dims(np.asarray(test_input), 0), test_input),  # do nothing
            np.expand_dims(np.asarray(test_input), 0),
        )

    def test_broadcast_to_axis_of(self):
        np.testing.assert_array_equal(reshaping.broadcast_to_axis_of(10, np.empty((2,)), 0), np.full(2, 10))
        assert reshaping.broadcast_to_axis_of(10, np.empty((2,)), 1) == 10
        np.testing.assert_array_equal(reshaping.broadcast_to_axis_of(10, np.empty((2, 3)), 0), np.full(2, 10))
        np.testing.assert_array_equal(reshaping.broadcast_to_axis_of(10, np.empty((2, 3)), 1), np.full(3, 10))
        assert reshaping.broadcast_to_axis_of(10, np.empty((2, 3)), 2) == 10

    def test_unstack_to_array(self):
        i = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4], ["a", "b", "c", "d"]])
        sr = pd.Series([1, 2, 3, 4], index=i)
        np.testing.assert_array_equal(
            reshaping.unstack_to_array(sr),
            np.asarray(
                [
                    [[1.0, np.nan, np.nan, np.nan], [np.nan, 2.0, np.nan, np.nan]],
                    [[np.nan, np.nan, 3.0, np.nan], [np.nan, np.nan, np.nan, 4.0]],
                ]
            ),
        )
        np.testing.assert_array_equal(reshaping.unstack_to_array(sr, levels=(0,)), np.array([2.0, 4.0]))
        np.testing.assert_array_equal(
            reshaping.unstack_to_array(sr, levels=(2, 0)),
            np.asarray(
                [
                    [1.0, np.nan],
                    [2.0, np.nan],
                    [np.nan, 3.0],
                    [np.nan, 4.0],
                ]
            ),
        )

    def test_make_symmetric(self):
        assert_frame_equal(
            reshaping.make_symmetric(sr2),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, 1.0, 2.0, 3.0],
                        [1.0, np.nan, np.nan, np.nan],
                        [2.0, np.nan, np.nan, np.nan],
                        [3.0, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.Index(["a2", "x2", "y2", "z2"], dtype="object", name=("i2", None)),
                columns=pd.Index(["a2", "x2", "y2", "z2"], dtype="object", name=("i2", None)),
            ),
        )
        assert_frame_equal(
            reshaping.make_symmetric(df2),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, 1.0, 2.0, 3.0],
                        [1.0, np.nan, np.nan, np.nan],
                        [2.0, np.nan, np.nan, np.nan],
                        [3.0, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.Index(["a4", "x4", "y4", "z4"], dtype="object", name=("i4", "c4")),
                columns=pd.Index(["a4", "x4", "y4", "z4"], dtype="object", name=("i4", "c4")),
            ),
        )
        assert_frame_equal(
            reshaping.make_symmetric(df5),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, 1.0, 4.0, 7.0],
                        [np.nan, np.nan, np.nan, 2.0, 5.0, 8.0],
                        [np.nan, np.nan, np.nan, 3.0, 6.0, 9.0],
                        [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                        [4.0, 5.0, 6.0, np.nan, np.nan, np.nan],
                        [7.0, 8.0, 9.0, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.MultiIndex.from_tuples(
                    [("a7", "a8"), ("b7", "b8"), ("c7", "c8"), ("x7", "x8"), ("y7", "y8"), ("z7", "z8")],
                    names=[("i7", "c7"), ("i8", "c8")],
                ),
                columns=pd.MultiIndex.from_tuples(
                    [("a7", "a8"), ("b7", "b8"), ("c7", "c8"), ("x7", "x8"), ("y7", "y8"), ("z7", "z8")],
                    names=[("i7", "c7"), ("i8", "c8")],
                ),
            ),
        )
        assert_frame_equal(
            reshaping.make_symmetric(pd.Series([1, 2, 3], name="yo"), sort=False),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, 1.0],
                        [np.nan, np.nan, np.nan, 2.0],
                        [np.nan, np.nan, np.nan, 3.0],
                        [1.0, 2.0, 3.0, np.nan],
                    ]
                ),
                index=pd.Index([0, 1, 2, "yo"], dtype="object"),
                columns=pd.Index([0, 1, 2, "yo"], dtype="object"),
            ),
        )

    def test_unstack_to_df(self):
        assert_frame_equal(
            reshaping.unstack_to_df(df5.iloc[0]),
            pd.DataFrame(
                np.array([[1.0, np.nan, np.nan], [np.nan, 2.0, np.nan], [np.nan, np.nan, 3.0]]),
                index=pd.Index(["a7", "b7", "c7"], dtype="object", name="c7"),
                columns=pd.Index(["a8", "b8", "c8"], dtype="object", name="c8"),
            ),
        )
        i = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4], ["a", "b", "c", "d"]])
        sr = pd.Series([1, 2, 3, 4], index=i)
        assert_frame_equal(
            reshaping.unstack_to_df(sr, index_levels=0, column_levels=1),
            pd.DataFrame(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                index=pd.Index([1, 2], dtype="int64"),
                columns=pd.Index([3, 4], dtype="int64"),
            ),
        )
        assert_frame_equal(
            reshaping.unstack_to_df(sr, index_levels=(0, 1), column_levels=2),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, np.nan, np.nan, np.nan],
                        [np.nan, 2.0, np.nan, np.nan],
                        [np.nan, np.nan, 3.0, np.nan],
                        [np.nan, np.nan, np.nan, 4.0],
                    ]
                ),
                index=pd.MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)]),
                columns=pd.Index(["a", "b", "c", "d"], dtype="object"),
            ),
        )
        assert_frame_equal(
            reshaping.unstack_to_df(sr, index_levels=0, column_levels=1, symmetric=True),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, 1.0, 2.0],
                        [np.nan, np.nan, 3.0, 4.0],
                        [1.0, 3.0, np.nan, np.nan],
                        [2.0, 4.0, np.nan, np.nan],
                    ]
                ),
                index=pd.Index([1, 2, 3, 4], dtype="int64"),
                columns=pd.Index([1, 2, 3, 4], dtype="int64"),
            ),
        )


# ############# indexing ############# #


called_dict = {}

PandasIndexer = indexing.PandasIndexer
ParamIndexer = indexing.build_param_indexer(["param1", "param2", "tuple"])


class H(PandasIndexer, ParamIndexer):
    def __init__(self, a, param1_mapper, param2_mapper, tuple_mapper, level_names):
        self.a = a

        self._param1_mapper = param1_mapper
        self._param2_mapper = param2_mapper
        self._tuple_mapper = tuple_mapper
        self._level_names = level_names

        PandasIndexer.__init__(self, calling="PandasIndexer")
        ParamIndexer.__init__(
            self,
            [param1_mapper, param2_mapper, tuple_mapper],
            level_names=[level_names[0], level_names[1], level_names],
            calling="ParamIndexer",
        )

    def indexing_func(self, pd_indexing_func, calling=None):
        # As soon as you call iloc etc., performs it on each dataframe and mapper and returns a new class instance
        called_dict[calling] = True
        param1_mapper = indexing.indexing_on_mapper(self._param1_mapper, self.a, pd_indexing_func)
        param2_mapper = indexing.indexing_on_mapper(self._param2_mapper, self.a, pd_indexing_func)
        tuple_mapper = indexing.indexing_on_mapper(self._tuple_mapper, self.a, pd_indexing_func)
        return H(pd_indexing_func(self.a), param1_mapper, param2_mapper, tuple_mapper, self._level_names)

    @classmethod
    def run(cls, a, params1, params2, level_names=("p1", "p2")):
        a = reshaping.to_2d(a)
        # Build column hierarchy
        params1_idx = pd.Index(params1, name=level_names[0])
        params2_idx = pd.Index(params2, name=level_names[1])
        params_idx = indexes.stack_indexes([params1_idx, params2_idx])
        new_columns = indexes.combine_indexes([params_idx, a.columns])

        # Build mappers
        param1_mapper = np.repeat(params1, len(a.columns))
        param1_mapper = pd.Series(param1_mapper, index=new_columns)

        param2_mapper = np.repeat(params2, len(a.columns))
        param2_mapper = pd.Series(param2_mapper, index=new_columns)

        tuple_mapper = list(zip(*list(map(lambda x: x.values, [param1_mapper, param2_mapper]))))
        tuple_mapper = pd.Series(tuple_mapper, index=new_columns)

        # Tile a to match the length of new_columns
        a = vbt.ArrayWrapper(a.index, new_columns, 2).wrap(reshaping.tile(a.values, 4, axis=1))
        return cls(a, param1_mapper, param2_mapper, tuple_mapper, level_names)


# Similate an indicator with two params
h = H.run(df4, [0.1, 0.1, 0.2, 0.2], [0.3, 0.4, 0.5, 0.6])


class TestIndexing:
    def test_kwargs(self):
        h[(0.1, 0.3, "a6")]
        assert called_dict["PandasIndexer"]
        h.param1_loc[0.1]
        assert called_dict["ParamIndexer"]

    def test_pandas_indexing(self):
        # __getitem__
        assert_series_equal(
            h[(0.1, 0.3, "a6")].a,
            pd.Series(
                np.array([1, 4, 7]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                name=(0.1, 0.3, "a6"),
            ),
        )
        # loc
        assert_frame_equal(
            h.loc[:, (0.1, 0.3, "a6"):(0.1, 0.3, "c6")].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, 0.3, "a6"), (0.1, 0.3, "b6"), (0.1, 0.3, "c6")],
                    names=["p1", "p2", "c6"],
                ),
            ),
        )
        # iloc
        assert_frame_equal(
            h.iloc[-2:, -2:].a,
            pd.DataFrame(
                np.array([[5, 6], [8, 9]]),
                index=pd.Index(["y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples([(0.2, 0.6, "b6"), (0.2, 0.6, "c6")], names=["p1", "p2", "c6"]),
            ),
        )
        # xs
        assert_frame_equal(
            h.xs((0.1, 0.3), level=("p1", "p2"), axis=1).a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
            ),
        )

    def test_param_indexing(self):
        # param1
        assert_frame_equal(
            h.param1_loc[0.1].a,
            pd.DataFrame(
                np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.3, "a6"), (0.3, "b6"), (0.3, "c6"), (0.4, "a6"), (0.4, "b6"), (0.4, "c6")],
                    names=["p2", "c6"],
                ),
            ),
        )
        # param2
        assert_frame_equal(
            h.param2_loc[0.3].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples([(0.1, "a6"), (0.1, "b6"), (0.1, "c6")], names=["p1", "c6"]),
            ),
        )
        # tuple
        assert_frame_equal(
            h.tuple_loc[(0.1, 0.3)].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
            ),
        )
        assert_frame_equal(
            h.tuple_loc[(0.1, 0.3):(0.1, 0.3)].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, 0.3, "a6"), (0.1, 0.3, "b6"), (0.1, 0.3, "c6")],
                    names=["p1", "p2", "c6"],
                ),
            ),
        )
        assert_frame_equal(
            h.tuple_loc[[(0.1, 0.3), (0.1, 0.3)]].a,
            pd.DataFrame(
                np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0.1, 0.3, "a6"),
                        (0.1, 0.3, "b6"),
                        (0.1, 0.3, "c6"),
                        (0.1, 0.3, "a6"),
                        (0.1, 0.3, "b6"),
                        (0.1, 0.3, "c6"),
                    ],
                    names=["p1", "p2", "c6"],
                ),
            ),
        )

    @pytest.mark.parametrize(
        "test_inputs",
        [(0, a1, a2, sr_none, sr1, sr2), (0, a1, a2, a3, a4, a5, sr_none, sr1, sr2, df_none, df1, df2, df3, df4)],
    )
    def test_flex(self, test_inputs):
        raw_args = reshaping.broadcast(
            *test_inputs,
            keep_flex=True,
            align_index=False,
        )
        bc_args = reshaping.broadcast(
            *test_inputs,
            keep_flex=False,
            align_index=False,
        )
        for r in range(len(test_inputs)):
            raw_arg = raw_args[r]
            bc_arg = np.array(bc_args[r])
            bc_arg_2d = reshaping.to_2d(bc_arg)
            for col in range(bc_arg_2d.shape[1]):
                for i in range(bc_arg_2d.shape[0]):
                    assert bc_arg_2d[i, col] == flex_indexing.flex_select_nb(raw_arg, i, col)

    def test_get_index_points(self):
        index = pd.date_range("2020-01-01", "2020-01-03", freq="3h", tz="+0400")
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every=2), np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
        )
        np.testing.assert_array_equal(indexing.get_index_points(index, every=2, start=5, end=10), np.array([5, 7, 9]))

        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h"), np.array([0, 2, 4, 5, 7, 9, 10, 12, 14, 15])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", add_delta="1h"), np.array([1, 2, 4, 6, 7, 9, 11, 12, 14, 16])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=5), np.array([5, 7, 9, 10, 12, 14, 15])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=5, end=12), np.array([5, 7, 9, 10, 12])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=index[5]), np.array([5, 7, 9, 10, 12, 14, 15])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=index[5], end=index[12]),
            np.array([5, 7, 9, 10, 12]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start="2020-01-01 15:00:00"),
            np.array([5, 7, 9, 10, 12, 14, 15]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start="2020-01-01 15:00:00", end="2020-01-02 12:00:00"),
            np.array([5, 7, 9, 10, 12]),
        )
        np.testing.assert_array_equal(indexing.get_index_points(index, at_time="12:00"), np.array([4, 12]))

        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
            ),
            np.array([0]),
        )
        np.testing.assert_array_equal(indexing.get_index_points(index, start=5), np.array([5]))
        np.testing.assert_array_equal(indexing.get_index_points(index, start=index[0]), np.array([0]))

        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5],
                end=index[10],
                kind="labels",
            ),
            np.array([5]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5] + pd.Timedelta(nanoseconds=1),
                end=index[10],
                kind="labels",
            ),
            np.array([6]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5] - pd.Timedelta(nanoseconds=1),
                end=index[10],
                kind="labels",
            ),
            np.array([5]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5],
                end=index[10] - pd.Timedelta(nanoseconds=1),
                kind="labels",
            ),
            np.array([5]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5],
                end=index[10] + pd.Timedelta(nanoseconds=1),
                kind="labels",
            ),
            np.array([5]),
        )

        np.testing.assert_array_equal(indexing.get_index_points(index, on=[5, 10]), np.array([5, 10]))
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=index[[5, 10]], at_time="12:00"), np.array([4, 12])
        )
        np.testing.assert_array_equal(indexing.get_index_points(index, on=[index[5], index[10]]), np.array([5, 10]))
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]], start=index[7]), np.array([10])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]], end=index[7]), np.array([5])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]], end=index[10]), np.array([5, 10])
        )
        np.testing.assert_array_equal(indexing.get_index_points(index, on=[index[5], index[10]], end=10), np.array([5]))

    def test_get_index_ranges(self):
        index = pd.date_range("2020-01-01", "2020-01-03", freq="3h", tz="+0400")
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=2, closed_start=False, closed_end=False)),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=2, closed_start=True, closed_end=False)),
            np.array([[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=2, closed_start=False, closed_end=True)),
            np.array([[1, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 13], [13, 15], [15, 17]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=2, closed_start=True, closed_end=True)),
            np.array([[0, 3], [2, 5], [4, 7], [6, 9], [8, 11], [10, 13], [12, 15], [14, 17]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, start=5, end=10, closed_start=False, closed_end=False)
            ),
            np.array([[6, 7], [8, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, start=5, end=10, closed_start=True, closed_end=False)
            ),
            np.array([[5, 7], [7, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, start=5, end=10, closed_start=False, closed_end=True)
            ),
            np.array([[6, 8], [8, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, start=5, end=10, closed_start=True, closed_end=True)
            ),
            np.array([[5, 8], [7, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=4, lookback_period=1)),
            np.array([[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=4, lookback_period=1, closed_end=True)),
            np.array([[0, 2], [4, 6], [8, 10], [12, 14]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=2, start=5, end=10, lookback_period=1)),
            np.array([[5, 6], [7, 8], [9, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=2, start=5, end=10, fixed_start=True)),
            np.array([[5, 7], [5, 9]]),
        )
        with pytest.raises(Exception):
            indexing.get_index_ranges(index, every=2, start=5, end=10, fixed_start=True, lookback_period=1)

        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", closed_start=False, closed_end=False)),
            np.array([[1, 2], [2, 4], [4, 5], [6, 7], [7, 9], [9, 10], [11, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", closed_start=True, closed_end=False)),
            np.array([[0, 2], [2, 4], [4, 5], [5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", closed_start=False, closed_end=True)),
            np.array([[1, 2], [2, 4], [4, 6], [6, 7], [7, 9], [9, 11], [11, 12], [12, 14], [14, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", closed_start=True, closed_end=True)),
            np.array([[0, 2], [2, 4], [4, 6], [5, 7], [7, 9], [9, 11], [10, 12], [12, 14], [14, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", add_start_delta="1h", add_end_delta="1h")),
            np.array([[1, 2], [2, 4], [4, 6], [6, 7], [7, 9], [9, 11], [11, 12], [12, 14], [14, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start=5)),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start=5, end=12)),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start=index[5])),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start=index[5], end=index[12])),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start="2020-01-01 15:00:00")),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", start="2020-01-01 15:00:00", end="2020-01-02 12:00:00")
            ),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", start="2020-01-01 15:00:00", fixed_start=True)
            ),
            np.array([[5, 7], [5, 9], [5, 10], [5, 12], [5, 14], [5, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", lookback_period="12h")),
            np.array([[0, 4], [2, 6], [4, 8], [5, 9], [7, 11], [9, 13], [10, 14], [12, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", lookback_period=4)),
            np.array([[0, 4], [2, 6], [4, 8], [5, 9], [7, 11], [9, 13], [10, 14], [12, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="12:00")),
            np.array([[4, 8], [12, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="12:00", end_time="15:00")),
            np.array([[4, 5], [12, 13]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end_time="15:00")),
            np.array([[0, 5], [8, 13], [16, 17]]),
        )
        assert len(np.column_stack(indexing.get_index_ranges(index, start_time="15:00", end_time="15:00"))) == 0
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="15:00", end_time="15:01")),
            np.array([[5, 6], [13, 14]], dtype=np.int_),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="15:00", end_time="14:59")),
            np.array([[5, 13], [13, 17]], dtype=np.int_),
        )

        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                )
            ),
            np.array([[0, 17]]),
        )
        np.testing.assert_array_equal(np.column_stack(indexing.get_index_ranges(index, start=5)), np.array([[5, 17]]))
        np.testing.assert_array_equal(np.column_stack(indexing.get_index_ranges(index, end=10)), np.array([[0, 10]]))
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=5, end=10)), np.array([[5, 10]])
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=index[0], end=index[10])),
            np.array([[0, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=5, end=[10, 15])),
            np.array([[5, 10], [5, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=[5, 7], end=10)),
            np.array([[5, 10], [7, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=index[5], end=[index[10], index[15]])),
            np.array([[5, 10], [5, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=[index[5], index[7]], end=index[10])),
            np.array([[5, 10], [7, 10]]),
        )
        with pytest.raises(Exception):
            indexing.get_index_ranges(index, start=0, end=index[10])
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=index[0], end=index[10], kind="labels")),
            np.array([[0, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end=10, lookback_period=3)), np.array([[7, 10]])
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end=index[10], lookback_period=3)),
            np.array([[7, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end=index[10], lookback_period="9h")),
            np.array([[7, 10]]),
        )

        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[9],
                    kind="labels",
                )
            ),
            np.array([[5, 8]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5] + pd.Timedelta(nanoseconds=1),
                    end=index[10],
                    kind="labels",
                )
            ),
            np.array([[6, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5] - pd.Timedelta(nanoseconds=1),
                    end=index[10],
                    kind="labels",
                )
            ),
            np.array([[5, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10] - pd.Timedelta(nanoseconds=1),
                    kind="labels",
                )
            ),
            np.array([[5, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10] + pd.Timedelta(nanoseconds=1),
                    kind="labels",
                )
            ),
            np.array([[5, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10],
                    kind="labels",
                    closed_start=False,
                )
            ),
            np.array([[6, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5] + pd.Timedelta(nanoseconds=1),
                    end=index[10],
                    kind="labels",
                    closed_start=False,
                )
            ),
            np.array([[6, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10],
                    kind="labels",
                    closed_end=True,
                )
            ),
            np.array([[5, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10] + pd.Timedelta(nanoseconds=1),
                    kind="labels",
                    closed_end=True,
                )
            ),
            np.array([[5, 10]]),
        )

    def test_get_idxs(self):
        i = pd.Index([1, 2, 3, 4, 5, 6], name="i")
        c = pd.Index([1, 2, 3, 4], name="c")
        dti = pd.date_range("2020-01-01", "2020-01-02", freq="1h", tz="Europe/Berlin")
        mi = pd.MultiIndex.from_arrays([[1, 1, 2, 2, 3, 3], [4, 5, 4, 5, 4, 5]], names=["i1", "i2"])
        mc = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4]], names=["c1", "c2"])

        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(0), i, c)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((0, 2)), i, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(((0, 2),)), i, c)
        np.testing.assert_array_equal(row_idxs, np.array([[0, 2]]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2)), i, c)
        assert row_idxs == slice(0, 2, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2, 2)), i, c)
        assert row_idxs == slice(0, 2, 2)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(None, None, None)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(0), dti, c)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((0, 2)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(((0, 2),)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([[0, 2]]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2)), dti, c)
        assert row_idxs == slice(0, 2, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2, 2)), dti, c)
        assert row_idxs == slice(0, 2, 2)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(0), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((0, 2)), i, c)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(0, 2)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 2, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(0, 2, 2)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 2, 2)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(None, None, None)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(1, kind="labels"), i, c)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((1, 3), kind="labels"), i, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(1, 3), kind="labels"), i, c)
        assert row_idxs == slice(0, 3, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(1, 3, 2), kind="labels"), i, c)
        assert row_idxs == slice(0, 3, 2)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 10), kind="labels"), i, c)
        assert row_idxs == slice(0, 6, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx("2020-01-01 17:00"), dti, c)
        assert row_idxs == 17
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(("2020-01-01", "2020-01-01 17:30")), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 18]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((1, 4)), mi, mc)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.rowidx((1, 3)), mi, mc)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice((1, 4), (2, 4))), mi, mc)
        assert row_idxs == slice(0, 3, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice((1, 4), (2, 4), 2)), mi, mc)
        assert row_idxs == slice(0, 3, 2)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(None, None, None)), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(1, kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((1, 3), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(1, 3), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(1, 3, 2), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, 2)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(0, 10), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 4, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((1, 3), kind="labels"), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((1, 2)), mi, mc)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((2, 5)), mi, mc)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(((1, 3), (2, 3))), mi, mc)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice((1, 3), (2, 3))), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice((1, 3), (2, 3), 2)), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, 2)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice((0, 0), (10, 10))), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 4, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(None, None, None)), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(1, level="c"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((1, 2), level="c"), i, c)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 1]))
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((1, 2, 0), level="c"), i, c)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(1, level="c1"), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 2, None)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx(0, level="c1"), mi, mc)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(3, level="c2"), mi, mc)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((3, 4), level="c2"), mi, mc)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2, 1, 3]))
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((3, 4, 0), level="c2"), mi, mc)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx(slice(0, 10), level="c2"), mi, mc)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(((1, 3), (2, 3)), level=("c1", "c2")), mi, mc)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))

        row_idxs, col_idxs = indexing.get_idxs(indexing.pointidx(on="2020"), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([0]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.pointidx(on=(1, 3, 5)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([1, 3, 5]))
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(
            indexing.rangeidx(start="2020-01-01 12:00", end="2020-01-01 17:00"), dti, c
        )
        np.testing.assert_array_equal(row_idxs, np.array([[12, 17]]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rangeidx(start=(0, 4), end=(2, 6)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([[0, 2], [4, 6]]))
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(
            indexing.idx(
                indexing.rowidx(1),
                indexing.colidx(2),
            ),
            i,
            c,
        )
        assert row_idxs == 1
        assert col_idxs == 2
        with pytest.raises(Exception):
            indexing.get_idxs(
                indexing.idx(
                    indexing.colidx(1),
                    indexing.colidx(2),
                ),
                i,
                c,
            )
        with pytest.raises(Exception):
            indexing.get_idxs(
                indexing.idx(
                    indexing.rowidx(1),
                    indexing.rowidx(2),
                ),
                i,
                c,
            )

        row_idxs, col_idxs = indexing.get_idxs(1, i, c, kind="labels")
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs((1, 3), i, c, kind="labels")
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(slice(1, 2), i, c, kind="labels")
        assert row_idxs == slice(0, 2, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs("2020-01-01 17:00", dti, c)
        assert row_idxs == 17
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(("2020-01-01", "2020-01-01 17:30"), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 18]))
        assert col_idxs == slice(None, None, None)

    def test_idx_setter_factory(self):
        idx_setter = indexing.IdxDict({0: 0, indexing.idx(1): 1}).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [(0, 0), (indexing.Idxr(1), 1)],
        )
        sr = pd.Series([3, 2, 1], index=["x", "y", "z"])
        idx_setter = indexing.IdxSeries(sr).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [(indexing.Idxr(sr.index), sr.values)],
        )
        idx_setter = indexing.IdxSeries(sr, split=True).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (indexing.Idxr(sr.index[0]), sr.values[0].item()),
                (indexing.Idxr(sr.index[1]), sr.values[1].item()),
                (indexing.Idxr(sr.index[2]), sr.values[2].item()),
            ],
        )
        df = pd.DataFrame([[6, 5], [4, 3], [2, 1]], index=["x", "y", "z"], columns=["a", "b"])
        idx_setter = indexing.IdxFrame(df).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [(indexing.Idxr(indexing.RowIdxr(df.index), indexing.ColIdxr(df.columns)), df.values)],
        )
        idx_setter = indexing.IdxFrame(df, split="columns").get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(df.index), indexing.ColIdxr(df.columns[0])), df.values[:, 0]),
                (indexing.Idxr(indexing.RowIdxr(df.index), indexing.ColIdxr(df.columns[1])), df.values[:, 1]),
            ],
        )
        idx_setter = indexing.IdxFrame(df, split="rows").get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(df.index[0]), indexing.ColIdxr(df.columns)), df.values[0]),
                (indexing.Idxr(indexing.RowIdxr(df.index[1]), indexing.ColIdxr(df.columns)), df.values[1]),
                (indexing.Idxr(indexing.RowIdxr(df.index[2]), indexing.ColIdxr(df.columns)), df.values[2]),
            ],
        )
        idx_setter = indexing.IdxFrame(df, split=True).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(df.index[0]), indexing.ColIdxr(df.columns[0])), df.values[0, 0].item()),
                (indexing.Idxr(indexing.RowIdxr(df.index[1]), indexing.ColIdxr(df.columns[0])), df.values[1, 0].item()),
                (indexing.Idxr(indexing.RowIdxr(df.index[2]), indexing.ColIdxr(df.columns[0])), df.values[2, 0].item()),
                (indexing.Idxr(indexing.RowIdxr(df.index[0]), indexing.ColIdxr(df.columns[1])), df.values[0, 1].item()),
                (indexing.Idxr(indexing.RowIdxr(df.index[1]), indexing.ColIdxr(df.columns[1])), df.values[1, 1].item()),
                (indexing.Idxr(indexing.RowIdxr(df.index[2]), indexing.ColIdxr(df.columns[1])), df.values[2, 1].item()),
            ],
        )
        records = pd.Series([3, 2, 1], index=["x", "y", "z"])
        idx_setters = indexing.IdxRecords(records).get()
        assert len(idx_setters) == 1
        assert checks.is_deep_equal(
            idx_setters["_1"].idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(records.index[0]), None), records.values[0].item()),
                (indexing.Idxr(indexing.RowIdxr(records.index[1]), None), records.values[1].item()),
                (indexing.Idxr(indexing.RowIdxr(records.index[2]), None), records.values[2].item()),
            ],
        )
        records = pd.DataFrame([[9, 8, 7], [6, 5, 4], [3, 2, 1]], columns=["row", "col", "X"])
        idx_setters = indexing.IdxRecords(records).get()
        assert len(idx_setters) == 1
        assert checks.is_deep_equal(
            idx_setters["X"].idx_items,
            [
                (
                    indexing.Idxr(
                        indexing.RowIdxr(records["row"].values[0].item()),
                        indexing.ColIdxr(records["col"].values[0].item()),
                    ),
                    records["X"].values[0].item(),
                ),
                (
                    indexing.Idxr(
                        indexing.RowIdxr(records["row"].values[1].item()),
                        indexing.ColIdxr(records["col"].values[1].item()),
                    ),
                    records["X"].values[1].item(),
                ),
                (
                    indexing.Idxr(
                        indexing.RowIdxr(records["row"].values[2].item()),
                        indexing.ColIdxr(records["col"].values[2].item()),
                    ),
                    records["X"].values[2].item(),
                ),
            ],
        )
        records1 = pd.DataFrame([[9, 8, 7], [6, 5, 4], [3, 2, 1]], columns=["row", "col", "X"])
        idx_setters1 = indexing.IdxRecords(records1).get()
        records2 = pd.DataFrame([[8, 7], [5, 4], [2, 1]], index=[9, 6, 3], columns=["col", "X"])
        idx_setters2 = indexing.IdxRecords(records2).get()
        assert checks.is_deep_equal(idx_setters1, idx_setters2)
        records3 = pd.DataFrame([[9, 8, 7], [6, 5, 4], [3, 2, 1]], index=[20, 30, 40], columns=["row", "col", "X"])
        with pytest.raises(Exception):
            indexing.IdxRecords(records3).get()
        idx_setters3 = indexing.IdxRecords(records3, row_field="row").get()
        assert checks.is_deep_equal(idx_setters1, idx_setters3)
        records = [
            dict(row=1, X=2, Y=3),
            dict(col=4, X=5),
            dict(row=6, col=7, Y=8),
            dict(Z=9),
        ]
        idx_setters = indexing.IdxRecords(records).get()
        assert len(idx_setters) == 3
        assert checks.is_deep_equal(
            idx_setters["X"].idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(1), None), 2),
                (indexing.Idxr(None, indexing.ColIdxr(4)), 5),
            ],
        )
        assert checks.is_deep_equal(
            idx_setters["Y"].idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(1), None), 3),
                (indexing.Idxr(indexing.RowIdxr(6), indexing.ColIdxr(7)), 8),
            ],
        )
        assert checks.is_deep_equal(
            idx_setters["Z"].idx_items,
            [
                ("_def", 9),
            ],
        )


# ############# flex_indexing ############# #


class TestFlexIndexing:
    def test_flex_select_nb(self):
        arr_1d = np.array([1, 2, 3])
        arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert flex_indexing.flex_select_1d_nb(arr_1d, 0) == arr_1d[0]
        assert flex_indexing.flex_select_1d_nb(arr_1d, 1) == arr_1d[1]
        with pytest.raises(Exception):
            flex_indexing.flex_select_nb(arr_1d, 100)
        assert flex_indexing.flex_select_1d_pr_nb(arr_1d, 100, rotate_rows=True) == arr_1d[100 % arr_1d.shape[0]]
        assert flex_indexing.flex_select_1d_pc_nb(arr_1d, 100, rotate_cols=True) == arr_1d[100 % arr_1d.shape[0]]
        assert flex_indexing.flex_select_nb(arr_2d, 0, 0) == arr_2d[0, 0]
        assert flex_indexing.flex_select_nb(arr_2d, 1, 0) == arr_2d[1, 0]
        assert flex_indexing.flex_select_nb(arr_2d, 0, 1) == arr_2d[0, 1]
        assert flex_indexing.flex_select_nb(arr_2d, 100, 0, rotate_rows=True) == arr_2d[100 % arr_2d.shape[0], 0]
        with pytest.raises(Exception):
            flex_indexing.flex_select_nb(arr_2d, 100, 0, rotate_rows=False)
        assert flex_indexing.flex_select_nb(arr_2d, 0, 100, rotate_cols=True) == arr_2d[0, 100 % arr_2d.shape[1]]
        with pytest.raises(Exception):
            flex_indexing.flex_select_nb(arr_2d, 0, 100, rotate_cols=False)


# ############# combining ############# #


class TestCombining:
    def test_apply_and_concat_none(self):
        def apply_func(i, x, a):
            x[i] = a[i]

        @njit
        def apply_func_nb(i, x, a):
            x[i] = a[i]

        # 1d
        target = pd.Series([10, 20, 30], index=sr2.index, name=sr2.name)
        sr2_copy = sr2.copy()
        combining.apply_and_concat(3, apply_func, sr2_copy.values, [10, 20, 30])
        assert_series_equal(sr2_copy, target)
        sr2_copy = sr2.copy()
        combining.apply_and_concat_none_nb(3, apply_func_nb, sr2_copy.values, (10, 20, 30))
        assert_series_equal(sr2_copy, target)
        sr2_copy = sr2.copy()
        combining.apply_and_concat(3, apply_func_nb, sr2_copy.values, [10, 20, 30], n_outputs=0, jitted_loop=True)
        assert_series_equal(sr2_copy, target)

    def test_apply_and_concat_one(self):
        def apply_func(i, x, a):
            return x + a[i]

        @njit
        def apply_func_nb(i, x, a):
            return x + a[i]

        # 1d
        target = np.array([[11, 21, 31], [12, 22, 32], [13, 23, 33]])
        np.testing.assert_array_equal(combining.apply_and_concat(3, apply_func, sr2.values, [10, 20, 30]), target)
        np.testing.assert_array_equal(
            combining.apply_and_concat_one_nb(3, apply_func_nb, sr2.values, (10, 20, 30)),
            target,
        )
        np.testing.assert_array_equal(
            combining.apply_and_concat(3, apply_func_nb, sr2.values, [10, 20, 30], n_outputs=1, jitted_loop=True),
            combining.apply_and_concat_one_nb(3, apply_func_nb, sr2.values, (10, 20, 30)),
        )
        # 2d
        target2 = np.array(
            [
                [11, 12, 13, 21, 22, 23, 31, 32, 33],
                [14, 15, 16, 24, 25, 26, 34, 35, 36],
                [17, 18, 19, 27, 28, 29, 37, 38, 39],
            ]
        )
        np.testing.assert_array_equal(combining.apply_and_concat(3, apply_func, df4.values, [10, 20, 30]), target2)
        np.testing.assert_array_equal(
            combining.apply_and_concat_one_nb(3, apply_func_nb, df4.values, (10, 20, 30)),
            target2,
        )
        np.testing.assert_array_equal(
            combining.apply_and_concat(3, apply_func_nb, df4.values, [10, 20, 30], n_outputs=1, jitted_loop=True),
            combining.apply_and_concat_one_nb(3, apply_func_nb, df4.values, (10, 20, 30)),
        )

    def test_apply_and_concat_multiple(self):
        def apply_func(i, x, a):
            return (x, x + a[i])

        @njit
        def apply_func_nb(i, x, a):
            return (x, x + a[i])

        # 1d
        target_a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        target_b = np.array([[11, 21, 31], [12, 22, 32], [13, 23, 33]])
        a, b = combining.apply_and_concat(3, apply_func, sr2.values, [10, 20, 30])
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat_multiple_nb(3, apply_func_nb, sr2.values, (10, 20, 30))
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat(3, apply_func_nb, sr2.values, [10, 20, 30], n_outputs=2, jitted_loop=True)
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        # 2d
        target_a = np.array([[1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9, 7, 8, 9]])
        target_b = np.array(
            [
                [11, 12, 13, 21, 22, 23, 31, 32, 33],
                [14, 15, 16, 24, 25, 26, 34, 35, 36],
                [17, 18, 19, 27, 28, 29, 37, 38, 39],
            ]
        )
        a, b = combining.apply_and_concat(3, apply_func, df4.values, [10, 20, 30])
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat_multiple_nb(3, apply_func_nb, df4.values, (10, 20, 30))
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat(3, apply_func_nb, df4.values, [10, 20, 30], n_outputs=2, jitted_loop=True)
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)

    def test_combine_and_concat(self):
        def combine_func(x, y, a):
            return x + y + a

        @njit
        def combine_func_nb(x, y, a):
            return x + y + a

        # 1d
        target = np.array([[103, 104], [106, 108], [109, 112]])
        np.testing.assert_array_equal(
            combining.combine_and_concat(sr2.values, (sr2.values * 2, sr2.values * 3), combine_func, 100),
            target,
        )
        np.testing.assert_array_equal(
            combining.combine_and_concat_nb(sr2.values, (sr2.values * 2, sr2.values * 3), combine_func_nb, 100),
            target,
        )
        # 2d
        target2 = np.array(
            [[103, 106, 109, 104, 108, 112], [112, 115, 118, 116, 120, 124], [121, 124, 127, 128, 132, 136]],
        )
        np.testing.assert_array_equal(
            combining.combine_and_concat(df4.values, (df4.values * 2, df4.values * 3), combine_func, 100),
            target2,
        )
        np.testing.assert_array_equal(
            combining.combine_and_concat_nb(df4.values, (df4.values * 2, df4.values * 3), combine_func_nb, 100),
            target2,
        )

    def test_combine_multiple(self):
        def combine_func(x, y, a):
            return x + y + a

        @njit
        def combine_func_nb(x, y, a):
            return x + y + a

        # 1d
        target = np.array([206, 212, 218])
        np.testing.assert_array_equal(
            combining.combine_multiple((sr2.values, sr2.values * 2, sr2.values * 3), combine_func, 100),
            target,
        )
        np.testing.assert_array_equal(
            combining.combine_multiple_nb((sr2.values, sr2.values * 2, sr2.values * 3), combine_func_nb, 100),
            target,
        )
        # 2d
        target2 = np.array([[206, 212, 218], [224, 230, 236], [242, 248, 254]])
        np.testing.assert_array_equal(
            combining.combine_multiple((df4.values, df4.values * 2, df4.values * 3), combine_func, 100),
            target2,
        )
        np.testing.assert_array_equal(
            combining.combine_multiple_nb((df4.values, df4.values * 2, df4.values * 3), combine_func_nb, 100),
            target2,
        )


# ############# merging ############# #


class TestMerging:
    def test_concat_merge(self):
        np.testing.assert_array_equal(
            merging.concat_merge([0, 1, 2]),
            np.array([0, 1, 2]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge(([0, 1, 2], [3, 4, 5])),
            np.array([0, 1, 2, 3, 4, 5]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge((([0, 1, 2],), ([0, 1, 2],)))[0],
            np.array([0, 1, 2, 0, 1, 2]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[0],
            np.array([0, 1, 2, 0, 1, 2]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[1],
            np.array([3, 4, 5, 3, 4, 5]),
        )
        assert_series_equal(
            merging.concat_merge([0, 1, 2], keys=pd.Index(["a", "b", "c"], name="d")),
            pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d")),
        )
        assert_series_equal(
            merging.concat_merge([0, 1, 2], wrap_kwargs=dict(index=pd.Index(["a", "b", "c"], name="d"), name="name")),
            pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d"), name="name"),
        )
        assert_series_equal(
            merging.concat_merge(([0, 1, 2], [3, 4, 5]), keys=pd.Index(["k1", "k2"], name="key")),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", 0), ("k1", 1), ("k1", 2), ("k2", 0), ("k2", 1), ("k2", 2)], names=["key", None]
                ),
            ),
        )
        assert_series_equal(
            merging.concat_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "a", "b", "c"], name="d"),
                name="name",
            ),
        )
        assert_series_equal(
            merging.concat_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=[
                    dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
                    dict(index=pd.Index(["e", "f", "g"], name="h"), name="name"),
                ],
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        sr1 = pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d"), name="name")
        sr2 = pd.Series([3, 4, 5], index=pd.Index(["e", "f", "g"], name="h"), name="name")
        assert_series_equal(
            merging.concat_merge((sr1, sr2)),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_series_equal(
            merging.concat_merge((sr1, sr2), keys=pd.Index(["k1", "k2"], name="key")),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")], names=["key", None]
                ),
                name="name",
            ),
        )
        assert_series_equal(
            merging.concat_merge([dict(a=0, b=1, c=2), dict(d=3, e=4, f=5)], wrap=True),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "d", "e", "f"]),
            ),
        )

    def test_row_stack_merge(self):
        np.testing.assert_array_equal(
            merging.row_stack_merge(([0, 1, 2], [3, 4, 5])),
            np.array([[0, 1, 2], [3, 4, 5]]),
        )
        np.testing.assert_array_equal(
            merging.row_stack_merge((([0, 1, 2],), ([0, 1, 2],)))[0],
            np.array([[0, 1, 2], [0, 1, 2]]),
        )
        np.testing.assert_array_equal(
            merging.row_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[0],
            np.array([[0, 1, 2], [0, 1, 2]]),
        )
        np.testing.assert_array_equal(
            merging.row_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[1],
            np.array([[3, 4, 5], [3, 4, 5]]),
        )
        assert_series_equal(
            merging.row_stack_merge(([0, 1, 2], [3, 4, 5]), keys=pd.Index(["k1", "k2"], name="key")),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", 0), ("k1", 1), ("k1", 2), ("k2", 0), ("k2", 1), ("k2", 2)], names=["key", None]
                ),
            ),
        )
        assert_series_equal(
            merging.row_stack_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "a", "b", "c"], name="d"),
                name="name",
            ),
        )
        assert_series_equal(
            merging.row_stack_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=[
                    dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
                    dict(index=pd.Index(["e", "f", "g"], name="h"), name="name"),
                ],
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge(
                ([[0], [1], [2]], [[3], [4], [5]]),
                wrap_kwargs=[
                    dict(index=pd.Index(["a", "b", "c"], name="d"), columns=["name"]),
                    dict(index=pd.Index(["e", "f", "g"], name="h"), columns=["name"]),
                ],
            ),
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                columns=["name"],
            ),
        )
        sr1 = pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d"), name="name")
        sr2 = pd.Series([3, 4, 5], index=pd.Index(["e", "f", "g"], name="h"), name="name")
        assert_series_equal(
            merging.row_stack_merge((sr1, sr2)),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_series_equal(
            merging.row_stack_merge((sr1, sr2), keys=pd.Index(["k1", "k2"], name="key")),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")], names=["key", None]
                ),
                name="name",
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge((sr1.to_frame(), sr2.to_frame())),
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                columns=["name"],
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge((sr1.to_frame(), sr2.to_frame()), keys=pd.Index(["k1", "k2"], name="key")),
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")], names=["key", None]
                ),
                columns=["name"],
            ),
        )
        assert_series_equal(
            merging.row_stack_merge([dict(a=0, b=1, c=2), dict(d=3, e=4, f=5)], wrap="sr"),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "d", "e", "f"]),
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge(
                [dict(a=[0], b=[1], c=[2]), dict(a=[3], b=[4], c=[5])],
                wrap="df",
                ignore_index=True,
            ),
            pd.DataFrame(
                [[0, 1, 2], [3, 4, 5]],
                columns=pd.Index(["a", "b", "c"]),
            ),
        )
        assert_series_equal(
            merging.row_stack_merge((sr1.vbt, sr2.vbt)).obj,
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_series_equal(
            merging.row_stack_merge((sr1.vbt, sr2.vbt), keys=pd.Index(["k1", "k2"], name="key")).obj,
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")], names=["key", None]
                ),
                name="name",
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge((sr1.to_frame().vbt, sr2.to_frame().vbt)).obj,
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                columns=["name"],
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge(
                (sr1.to_frame().vbt, sr2.to_frame().vbt), keys=pd.Index(["k1", "k2"], name="key")
            ).obj,
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")], names=["key", None]
                ),
                columns=["name"],
            ),
        )

    def test_column_stack_merge(self):
        np.testing.assert_array_equal(
            merging.column_stack_merge(([0, 1, 2], [3, 4, 5])),
            np.array([[0, 3], [1, 4], [2, 5]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((([0, 1, 2],), ([0, 1, 2],)))[0],
            np.array([[0, 0], [1, 1], [2, 2]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[0],
            np.array([[0, 0], [1, 1], [2, 2]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[1],
            np.array([[3, 3], [4, 4], [5, 5]]),
        )
        assert_frame_equal(
            merging.column_stack_merge(([[0, 1, 2]], [[3, 4, 5]]), keys=pd.Index(["k1", "k2"], name="key")),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.MultiIndex.from_tuples(
                    [("k1", 0), ("k1", 1), ("k1", 2), ("k2", 0), ("k2", 1), ("k2", 2)], names=["key", None]
                ),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge(
                ([[0, 1, 2]], [[3, 4, 5]]),
                wrap_kwargs=dict(columns=pd.Index(["a", "b", "c"]), index=pd.Index(["d"])),
            ),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "a", "b", "c"]),
                index=pd.Index(["d"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge(
                ([[0, 1, 2]], [[3, 4, 5]]),
                wrap_kwargs=[
                    dict(columns=pd.Index(["a", "b", "c"], name="d")),
                    dict(columns=pd.Index(["e", "f", "g"], name="h")),
                ],
            ),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "e", "f", "g"]),
            ),
        )
        df1 = pd.DataFrame([[0, 1, 2]], columns=pd.Index(["a", "b", "c"], name="d"), index=pd.Index(["i"]))
        df2 = pd.DataFrame([[3, 4, 5]], columns=pd.Index(["e", "f", "g"], name="h"), index=pd.Index(["i"]))
        assert_frame_equal(
            merging.column_stack_merge((df1, df2)),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "e", "f", "g"]),
                index=pd.Index(["i"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((df1, df2), keys=pd.Index(["k1", "k2"], name="key")),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")], names=["key", None]
                ),
                index=pd.Index(["i"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge([dict(a=0, b=1, c=2), dict(a=3, b=4, c=5)], wrap="sr"),
            pd.DataFrame(
                [[0, 3], [1, 4], [2, 5]],
                index=pd.Index(["a", "b", "c"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge([dict(a=[0], b=[1], c=[2]), dict(d=[3], e=[4], f=[5])], wrap="df"),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "d", "e", "f"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((df1.vbt, df2.vbt)).obj,
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "e", "f", "g"]),
                index=pd.Index(["i"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((df1.vbt, df2.vbt), keys=pd.Index(["k1", "k2"], name="key")).obj,
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")], names=["key", None]
                ),
                index=pd.Index(["i"]),
            ),
        )
        sr1 = pd.Series([0, 1, 2, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        sr2 = pd.Series([6, 7, 8], index=pd.date_range("2020-01-04", periods=3))
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index=False),
            pd.DataFrame(
                [[0.0, np.nan], [1.0, np.nan], [2.0, np.nan], [4.0, 6.0], [5.0, 7.0], [np.nan, 8.0]],
                index=pd.date_range("2020-01-01", periods=6),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index=True),
            pd.DataFrame([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index="from_start"),
            pd.DataFrame([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index="from_end"),
            pd.DataFrame([[0, np.nan], [1, np.nan], [2, 6.0], [4, 7.0], [5, 8.0]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index=False).obj,
            pd.DataFrame(
                [[0.0, np.nan], [1.0, np.nan], [2.0, np.nan], [4.0, 6.0], [5.0, 7.0], [np.nan, 8.0]],
                index=pd.date_range("2020-01-01", periods=6),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index=True).obj,
            pd.DataFrame([[0.0, 6.0], [1.0, 7.0], [2.0, 8.0], [4.0, np.nan], [5.0, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index="from_start").obj,
            pd.DataFrame([[0.0, 6.0], [1.0, 7.0], [2.0, 8.0], [4.0, np.nan], [5.0, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index="from_end").obj,
            pd.DataFrame([[0.0, np.nan], [1.0, np.nan], [2.0, 6.0], [4.0, 7.0], [5.0, 8.0]]),
        )
        arr1 = np.array([0, 1, 2, 4, 5])
        arr2 = np.array([6, 7, 8])
        with pytest.raises(Exception):
            merging.column_stack_merge((arr1, arr2), reset_index=False)
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index=True),
            np.array([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_start"),
            np.array([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_end"),
            np.array([[0, np.nan], [1, np.nan], [2, 6.0], [4, 7.0], [5, 8.0]]),
        )
        arr1 = np.array([0, 1, 2])
        arr2 = np.array([6, 7, 8])
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index=False),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index=True),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_start"),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_end"),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )

    def test_mixed_merge(self):
        np.testing.assert_array_equal(
            merging.mixed_merge(
                [
                    ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                    ([9, 10, 11], [12, 13, 14], [15, 16, 17]),
                ],
                func_names=("concat", "row_stack", "column_stack"),
            )[0],
            np.array([0, 1, 2, 9, 10, 11]),
        )
        np.testing.assert_array_equal(
            merging.mixed_merge(
                [
                    ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                    ([9, 10, 11], [12, 13, 14], [15, 16, 17]),
                ],
                func_names=("concat", "row_stack", "column_stack"),
            )[1],
            np.array([[3, 4, 5], [12, 13, 14]]),
        )
        np.testing.assert_array_equal(
            merging.mixed_merge(
                [
                    ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                    ([9, 10, 11], [12, 13, 14], [15, 16, 17]),
                ],
                func_names=("concat", "row_stack", "column_stack"),
            )[2],
            np.array([[6, 15], [7, 16], [8, 17]]),
        )
        np.testing.assert_array_equal(
            merging.resolve_merge_func(("concat", "row_stack", "column_stack"))(
                [
                    ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                    ([9, 10, 11], [12, 13, 14], [15, 16, 17]),
                ]
            )[0],
            np.array([0, 1, 2, 9, 10, 11]),
        )


# ############# accessors ############# #


class TestAccessors:
    def test_indexing(self):
        assert_series_equal(df4.vbt["a6"].obj, df4["a6"].vbt.obj)

    def test_freq(self):
        ts = pd.Series(
            [1, 2, 3],
            index=pd.DatetimeIndex([datetime(2018, 1, 1), datetime(2018, 1, 2), datetime(2018, 1, 3)]),
        )
        assert ts.vbt.wrapper.freq == day_dt
        assert ts.vbt(freq="2D").wrapper.freq == day_dt * 2
        assert pd.Series([1, 2, 3]).vbt.wrapper.freq is None
        assert pd.Series([1, 2, 3]).vbt(freq="3D").wrapper.freq == day_dt * 3
        assert pd.Series([1, 2, 3]).vbt(freq=np.timedelta64(4, "D")).wrapper.freq == day_dt * 4

    def test_props(self):
        assert sr1.vbt.is_series()
        assert not sr1.vbt.is_frame()
        assert not df1.vbt.is_series()
        assert df2.vbt.is_frame()

    def test_wrapper(self):
        assert_index_equal(sr2.vbt.wrapper.index, sr2.index)
        assert_index_equal(sr2.vbt.wrapper.columns, sr2.to_frame().columns)
        assert sr2.vbt.wrapper.ndim == sr2.ndim
        assert sr2.vbt.wrapper.name == sr2.name
        assert pd.Series([1, 2, 3]).vbt.wrapper.name is None
        assert sr2.vbt.wrapper.shape == sr2.shape
        assert sr2.vbt.wrapper.shape_2d == (sr2.shape[0], 1)
        assert_index_equal(df4.vbt.wrapper.index, df4.index)
        assert_index_equal(df4.vbt.wrapper.columns, df4.columns)
        assert df4.vbt.wrapper.ndim == df4.ndim
        assert df4.vbt.wrapper.name is None
        assert df4.vbt.wrapper.shape == df4.shape
        assert df4.vbt.wrapper.shape_2d == df4.shape
        assert_series_equal(sr2.vbt.wrapper.wrap(a2), sr2)
        assert_series_equal(sr2.vbt.wrapper.wrap(df2), sr2)
        assert_series_equal(
            sr2.vbt.wrapper.wrap(df2.values, index=df2.index, columns=df2.columns),
            pd.Series(df2.values[:, 0], index=df2.index, name=df2.columns[0]),
        )
        assert_frame_equal(
            sr2.vbt.wrapper.wrap(df4.values, columns=df4.columns),
            pd.DataFrame(df4.values, index=sr2.index, columns=df4.columns),
        )
        assert_frame_equal(df2.vbt.wrapper.wrap(a2), df2)
        assert_frame_equal(df2.vbt.wrapper.wrap(sr2), df2)
        assert_frame_equal(
            df2.vbt.wrapper.wrap(df4.values, columns=df4.columns),
            pd.DataFrame(df4.values, index=df2.index, columns=df4.columns),
        )

    def test_row_stack(self):
        acc = vbt.BaseAccessor.row_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"]))),
            vbt.BaseAccessor(pd.Series([3, 4, 5], index=pd.Index(["d", "e", "f"]))),
        )
        target_obj = pd.DataFrame(
            [0, 1, 2, 3, 4, 5],
            index=pd.Index(["a", "b", "c", "d", "e", "f"]),
        )
        assert isinstance(acc, vbt.BaseSRAccessor)
        assert_index_equal(
            acc.wrapper.index,
            target_obj.index,
        )
        assert acc.wrapper.name is None
        assert target_obj.vbt.wrapper.ndim == target_obj.ndim
        acc = vbt.BaseAccessor.row_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
            vbt.BaseAccessor(pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))),
        )
        target_obj = pd.DataFrame(
            [[0, 0], [1, 1], [2, 2], [0, 1], [2, 3], [4, 5]],
            index=pd.RangeIndex(start=0, stop=6, step=1),
            columns=pd.Index(["a", "b"], dtype="object", name="c"),
        )
        assert isinstance(acc, vbt.BaseDFAccessor)
        assert_index_equal(
            acc.wrapper.index,
            target_obj.index,
        )
        assert_index_equal(
            acc.wrapper.columns,
            target_obj.columns,
        )
        assert target_obj.vbt.wrapper.ndim == target_obj.ndim
        acc = vbt.BaseAccessor.row_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False),
            vbt.BaseAccessor(
                pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                some_arg=2,
                check_expected_keys_=False,
            ),
        )
        assert acc.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.BaseAccessor.row_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False),
                vbt.BaseAccessor(pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.row_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=2,
                    check_expected_keys_=False,
                ),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.row_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=3,
                    check_expected_keys_=False,
                ),
            )

    def test_column_stack(self):
        acc = vbt.BaseAccessor.column_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
            vbt.BaseAccessor(pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))),
        )
        target_obj = pd.DataFrame(
            [[0, 0, 1], [1, 2, 3], [2, 4, 5]],
            index=pd.RangeIndex(start=0, stop=3, step=1),
            columns=pd.Index(["sr", "a", "b"], dtype="object"),
        )
        assert isinstance(acc, vbt.BaseDFAccessor)
        assert_index_equal(
            acc.wrapper.index,
            target_obj.index,
        )
        assert_index_equal(
            acc.wrapper.columns,
            target_obj.columns,
        )
        acc = vbt.BaseAccessor.column_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False),
            vbt.BaseAccessor(
                pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                some_arg=2,
                check_expected_keys_=False,
            ),
        )
        assert acc.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.BaseAccessor.column_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False),
                vbt.BaseAccessor(pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.column_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=2,
                    check_expected_keys_=False,
                ),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.column_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=3,
                    check_expected_keys_=False,
                ),
            )

    def test_empty(self):
        assert_series_equal(
            pd.Series.vbt.empty(5, index=np.arange(10, 15), name="a", fill_value=5),
            pd.Series(np.full(5, 5), index=np.arange(10, 15), name="a"),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.empty((5, 3), index=np.arange(10, 15), columns=["a", "b", "c"], fill_value=5),
            pd.DataFrame(np.full((5, 3), 5), index=np.arange(10, 15), columns=["a", "b", "c"]),
        )
        assert_series_equal(
            pd.Series.vbt.empty_like(sr2, fill_value=5),
            pd.Series(np.full(sr2.shape, 5), index=sr2.index, name=sr2.name),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.empty_like(df4, fill_value=5),
            pd.DataFrame(np.full(df4.shape, 5), index=df4.index, columns=df4.columns),
        )

    def test_apply_func_on_index(self):
        assert_frame_equal(
            df1.vbt.apply_on_index(lambda idx: idx + "_yo", axis=0),
            pd.DataFrame(
                np.array([1]),
                index=pd.Index(["x3_yo"], dtype="object", name="i3"),
                columns=pd.Index(["a3"], dtype="object", name="c3"),
            ),
        )
        assert_frame_equal(
            df1.vbt.apply_on_index(lambda idx: idx + "_yo", axis=1),
            pd.DataFrame(
                np.array([1]),
                index=pd.Index(["x3"], dtype="object", name="i3"),
                columns=pd.Index(["a3_yo"], dtype="object", name="c3"),
            ),
        )
        df1_copy = df1.vbt.apply_on_index(lambda idx: idx + "_yo", axis=0, copy_data=True)
        df1_copy.iloc[0, 0] = -1
        assert df1.iloc[0, 0] == 1
        df1_copy2 = df1.vbt.apply_on_index(lambda idx: idx + "_yo", axis=1, copy_data=True)
        df1_copy2.iloc[0, 0] = -1
        assert df1.iloc[0, 0] == 1

    def test_stack_index(self):
        assert_frame_equal(
            df5.vbt.stack_index([1, 2, 3], on_top=True),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.MultiIndex.from_tuples(
                    [(1, "a7", "a8"), (2, "b7", "b8"), (3, "c7", "c8")],
                    names=[None, "c7", "c8"],
                ),
            ),
        )
        assert_frame_equal(
            df5.vbt.stack_index([1, 2, 3], on_top=False),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a7", "a8", 1), ("b7", "b8", 2), ("c7", "c8", 3)],
                    names=["c7", "c8", None],
                ),
            ),
        )

    def test_drop_levels(self):
        assert_frame_equal(
            df5.vbt.drop_levels("c7"),
            pd.DataFrame(df5.values, index=df5.index, columns=pd.Index(["a8", "b8", "c8"], dtype="object", name="c8")),
        )

    def test_rename_levels(self):
        assert_frame_equal(
            df5.vbt.rename_levels({"c8": "c9"}),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.MultiIndex.from_tuples([("a7", "a8"), ("b7", "b8"), ("c7", "c8")], names=["c7", "c9"]),
            ),
        )

    def test_select_levels(self):
        assert_frame_equal(
            df5.vbt.select_levels("c8"),
            pd.DataFrame(df5.values, index=df5.index, columns=pd.Index(["a8", "b8", "c8"], dtype="object", name="c8")),
        )

    def test_drop_redundant_levels(self):
        assert_frame_equal(
            df5.vbt.stack_index(pd.RangeIndex(start=0, step=1, stop=3)).vbt.drop_redundant_levels(),
            df5,
        )

    def test_drop_duplicate_levels(self):
        assert_frame_equal(
            df5.vbt.stack_index(df5.columns.get_level_values(0)).vbt.drop_duplicate_levels(),
            df5,
        )

    def test_set(self):
        ts_index = pd.date_range("2020-01-01", "2020-01-05")
        df = pd.DataFrame(0, index=ts_index, columns=["a", "b", "c"])
        sr = pd.Series(0, index=ts_index)

        target_sr = sr.copy()
        target_sr.iloc[::2] = 100
        assert_series_equal(sr.vbt.set(100, every=2), target_sr)
        target_df = df.copy()
        target_df.iloc[::2] = 100
        assert_frame_equal(df.vbt.set(100, every=2), target_df)
        target_df = df.copy()
        target_df.iloc[::2, 1] = 100
        assert_frame_equal(df.vbt.set(100, columns="b", every=2), target_df)
        target_df = df.copy()
        target_df.iloc[::2, [1, 2]] = 100
        assert_frame_equal(df.vbt.set(100, columns=["b", "c"], every=2), target_df)

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(sr.vbt.set([100, 200, 300], every=2), target_sr)
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(df.vbt.set([100, 200, 300], every=2), target_df)
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(df.vbt.set([100, 200, 300], columns="b", every=2), target_df)
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(df.vbt.set([100, 200, 300], columns=["b", "c"], every=2), target_df)

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(
            sr.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), every=2),
            target_sr,
        )
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(
            df.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), every=2),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(
            df.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), columns="b", every=2),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(
            df.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), columns=["b", "c"], every=2),
            target_df,
        )

    def test_set_between(self):
        ts_index = pd.date_range("2020-01-01", "2020-01-05")
        df = pd.DataFrame(0, index=ts_index, columns=["a", "b", "c"])
        sr = pd.Series(0, index=ts_index)

        target_sr = sr.copy()
        target_sr.iloc[::2] = 100
        assert_series_equal(sr.vbt.set_between(100, start=[0, 2, 4], end=[1, 3, 5]), target_sr)
        target_df = df.copy()
        target_df.iloc[::2] = 100
        assert_frame_equal(df.vbt.set_between(100, start=[0, 2, 4], end=[1, 3, 5]), target_df)
        target_df = df.copy()
        target_df.iloc[::2, 1] = 100
        assert_frame_equal(
            df.vbt.set_between(100, columns="b", start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[::2, [1, 2]] = 100
        assert_frame_equal(
            df.vbt.set_between(100, columns=["b", "c"], start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(sr.vbt.set_between([100, 200, 300], start=[0, 2, 4], end=[1, 3, 5]), target_sr)
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(df.vbt.set_between([100, 200, 300], start=[0, 2, 4], end=[1, 3, 5]), target_df)
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(
            df.vbt.set_between([100, 200, 300], columns="b", start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(
            df.vbt.set_between([100, 200, 300], columns=["b", "c"], start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(
            sr.vbt.set_between(lambda i: [100, 200, 300][i], vbt.Rep("i"), start=[0, 2, 4], end=[1, 3, 5]),
            target_sr,
        )
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(
            df.vbt.set_between(lambda i: [100, 200, 300][i], vbt.Rep("i"), start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(
            df.vbt.set_between(lambda i: [100, 200, 300][i], vbt.Rep("i"), columns="b", start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(
            df.vbt.set_between(
                lambda i: [100, 200, 300][i],
                vbt.Rep("i"),
                columns=["b", "c"],
                start=[0, 2, 4],
                end=[1, 3, 5],
            ),
            target_df,
        )

    def test_to_array(self):
        np.testing.assert_array_equal(sr2.vbt.to_1d_array(), sr2.values)
        np.testing.assert_array_equal(sr2.vbt.to_2d_array(), sr2.to_frame().values)
        np.testing.assert_array_equal(df2.vbt.to_1d_array(), df2.iloc[:, 0].values)
        np.testing.assert_array_equal(df2.vbt.to_2d_array(), df2.values)

    def test_tile(self):
        assert_frame_equal(
            df4.vbt.tile(2, keys=["a", "b"], axis=0),
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.MultiIndex.from_tuples(
                    [("a", "x6"), ("a", "y6"), ("a", "z6"), ("b", "x6"), ("b", "y6"), ("b", "z6")],
                    names=[None, "i6"],
                ),
                columns=df4.columns,
            ),
        )
        assert_frame_equal(
            df4.vbt.tile(2, keys=["a", "b"], axis=1),
            pd.DataFrame(
                np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]),
                index=df4.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a6"), ("a", "b6"), ("a", "c6"), ("b", "a6"), ("b", "b6"), ("b", "c6")],
                    names=[None, "c6"],
                ),
            ),
        )

    def test_repeat(self):
        assert_frame_equal(
            df4.vbt.repeat(2, keys=["a", "b"], axis=0),
            pd.DataFrame(
                np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [7, 8, 9], [7, 8, 9]]),
                index=pd.MultiIndex.from_tuples(
                    [("x6", "a"), ("x6", "b"), ("y6", "a"), ("y6", "b"), ("z6", "a"), ("z6", "b")],
                    names=["i6", None],
                ),
                columns=df4.columns,
            ),
        )
        assert_frame_equal(
            df4.vbt.repeat(2, keys=["a", "b"], axis=1),
            pd.DataFrame(
                np.array([[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6], [7, 7, 8, 8, 9, 9]]),
                index=df4.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a6", "a"), ("a6", "b"), ("b6", "a"), ("b6", "b"), ("c6", "a"), ("c6", "b")],
                    names=["c6", None],
                ),
            ),
        )

    def test_align_to(self):
        multi_c1 = pd.MultiIndex.from_arrays([["a8", "b8"]], names=["c8"])
        multi_c2 = pd.MultiIndex.from_arrays([["a7", "a7", "c7", "c7"], ["a8", "b8", "a8", "b8"]], names=["c7", "c8"])
        df10 = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=multi_c1)
        df20 = pd.DataFrame([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], columns=multi_c2)
        assert_frame_equal(
            df10.vbt.align_to(df20),
            pd.DataFrame(
                np.array([[1, 2, 1, 2], [4, 5, 4, 5], [7, 8, 7, 8]]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=multi_c2,
            ),
        )

    def test_broadcast(self):
        a, b = pd.Series.vbt.broadcast(sr2, 10)
        b_target = pd.Series(np.full(sr2.shape, 10), index=sr2.index, name=sr2.name)
        assert_series_equal(a, sr2)
        assert_series_equal(b, b_target)
        a, b = sr2.vbt.broadcast(10)
        assert_series_equal(a, sr2)
        assert_series_equal(b, b_target)

    def test_broadcast_to(self):
        assert_frame_equal(sr2.vbt.broadcast_to(df2, align_index=False), df2)
        assert_frame_equal(sr2.vbt.broadcast_to(df2.vbt, align_index=False), df2)

    def test_broadcast_combs(self):
        new_index = pd.MultiIndex.from_tuples(
            [("x6", "x7", "x8"), ("y6", "y7", "y8"), ("z6", "z7", "z8")],
            names=["i6", "i7", "i8"],
        )
        new_columns = pd.MultiIndex.from_tuples(
            [
                ("a6", "a7", "a8"),
                ("a6", "b7", "b8"),
                ("a6", "c7", "c8"),
                ("b6", "a7", "a8"),
                ("b6", "b7", "b8"),
                ("b6", "c7", "c8"),
                ("c6", "a7", "a8"),
                ("c6", "b7", "b8"),
                ("c6", "c7", "c8"),
            ],
            names=["c6", "c7", "c8"],
        )
        assert_frame_equal(
            df4.vbt.broadcast_combs(df5, align_index=False)[0],
            pd.DataFrame(
                [[1, 1, 1, 2, 2, 2, 3, 3, 3], [4, 4, 4, 5, 5, 5, 6, 6, 6], [7, 7, 7, 8, 8, 8, 9, 9, 9]],
                index=new_index,
                columns=new_columns,
            ),
        )
        assert_frame_equal(
            df4.vbt.broadcast_combs(df5, align_index=False)[1],
            pd.DataFrame(
                [[1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9, 7, 8, 9]],
                index=new_index,
                columns=new_columns,
            ),
        )

    def test_apply(self):
        assert_series_equal(sr2.vbt.apply(lambda x: x**2), sr2**2)
        assert_series_equal(sr2.vbt.apply(lambda x: x**2, to_2d=True), sr2**2)
        assert_frame_equal(df4.vbt.apply(lambda x: x**2), df4**2)
        assert_frame_equal(
            sr2.vbt.apply(lambda x, y: x**y, vbt.Rep("y"), broadcast_named_args=dict(y=df4)),
            sr2.vbt**df4,
        )

    def test_concat(self):
        assert_frame_equal(
            pd.DataFrame.vbt.concat(
                pd.Series([1, 2, 3]),
                pd.Series([1, 2, 3]),
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame({0: pd.Series([1, 2, 3]), 1: pd.Series([1, 2, 3])}),
        )
        target = pd.DataFrame(
            np.array([[1, 1, 1, 10, 10, 10, 1, 2, 3], [2, 2, 2, 10, 10, 10, 4, 5, 6], [3, 3, 3, 10, 10, 10, 7, 8, 9]]),
            index=pd.MultiIndex.from_tuples([("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]),
            columns=pd.MultiIndex.from_tuples(
                [
                    ("a", "a6"),
                    ("a", "b6"),
                    ("a", "c6"),
                    ("b", "a6"),
                    ("b", "b6"),
                    ("b", "c6"),
                    ("c", "a6"),
                    ("c", "b6"),
                    ("c", "c6"),
                ],
                names=[None, "c6"],
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.concat(sr2, 10, df4, keys=["a", "b", "c"], broadcast_kwargs=dict(align_index=False)),
            target,
        )
        assert_frame_equal(
            sr2.vbt.concat(10, df4, keys=["a", "b", "c"], broadcast_kwargs=dict(align_index=False)),
            target,
        )

    def test_apply_and_concat(self):
        def apply_func(i, x, y, c, d=1):
            return x + y[i] + c + d

        @njit
        def apply_func_nb(i, x, y, c, d):
            return x + y[i] + c + d

        target = pd.DataFrame(
            np.array([[112, 113, 114], [113, 114, 115], [114, 115, 116]]),
            index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"),
            columns=pd.Index(["a", "b", "c"], dtype="object"),
        )
        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func_nb,
                np.array([1, 2, 3]),
                10,
                100,
                jitted_loop=True,
                n_outputs=1,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func,
                np.array([1, 2, 3]),
                10,
                d=100,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                target.values,
                index=target.index,
                columns=pd.Index([0, 1, 2], dtype="int64", name="apply_idx"),
            ),
        )

        def apply_func2(i, x, y, c, d=1):
            return x + y + c + d

        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func2,
                np.array([[1], [2], [3]]),
                10,
                d=100,
                keys=["a", "b", "c"],
                to_2d=True,  # otherwise (3, 1) + (1, 3) = (3, 3) != (3, 1) -> error
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                np.array([[112, 112, 112], [114, 114, 114], [116, 116, 116]]),
                index=target.index,
                columns=target.columns,
            ),
        )
        target2 = pd.DataFrame(
            np.array([[112, 113, 114], [113, 114, 115], [114, 115, 116]]),
            index=pd.Index(["x4", "y4", "z4"], dtype="object", name="i4"),
            columns=pd.MultiIndex.from_tuples([("a", "a4"), ("b", "a4"), ("c", "a4")], names=[None, "c4"]),
        )
        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )
        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func_nb,
                np.array([1, 2, 3]),
                10,
                100,
                jitted_loop=True,
                n_outputs=1,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )

        def apply_func3(i, x, y, c, d=1):
            return (x + y[i] + c + d, x + y[i] + c + d)

        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func3,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            )[0],
            target2,
        )
        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func3,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            )[1],
            target2,
        )

        def apply_func2(i, x, y, z):
            return x + y + z[i]

        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func2,
                vbt.Rep("y"),
                vbt.RepEval("np.arange(ntimes)"),
                broadcast_named_args=dict(y=df4),
                template_context=dict(np=np),
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                [[2, 3, 4, 3, 4, 5, 4, 5, 6], [6, 7, 8, 7, 8, 9, 8, 9, 10], [10, 11, 12, 11, 12, 13, 12, 13, 14]],
                index=pd.MultiIndex.from_tuples([("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]),
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a6"), (0, "b6"), (0, "c6"), (1, "a6"), (1, "b6"), (1, "c6"), (2, "a6"), (2, "b6"), (2, "c6")],
                    names=["apply_idx", "c6"],
                ),
            ),
        )

    def test_combine(self):
        def combine_func(x, y, a, b=1):
            return x + y + a + b

        @njit
        def combine_func_nb(x, y, a, b):
            return x + y + a + b

        assert_series_equal(
            sr2.vbt.combine(
                10,
                combine_func,
                100,
                b=1000,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.Series(
                np.array([1111, 1112, 1113]),
                index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"),
                name=sr2.name,
            ),
        )
        assert_series_equal(
            sr2.vbt.combine(
                10,
                combine_func,
                100,
                1000,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.Series(
                np.array([1111, 1112, 1113]),
                index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"),
                name=sr2.name,
            ),
        )

        @njit
        def combine_func2_nb(x, y):
            return x + y + np.array([[1], [2], [3]])

        assert_series_equal(
            sr2.vbt.combine(
                10,
                combine_func2_nb,
                to_2d=True,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.Series(np.array([12, 14, 16]), index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"), name="a2"),
        )

        @njit
        def combine_func3_nb(x, y):
            return x + y

        assert_frame_equal(
            df4.vbt.combine(
                sr2,
                combine_func3_nb,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                np.array([[2, 3, 4], [6, 7, 8], [10, 11, 12]]),
                index=pd.MultiIndex.from_tuples([("x6", "x2"), ("y6", "y2"), ("z6", "z2")], names=["i6", "i2"]),
                columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.combine(
                [df4, sr2],
                combine_func3_nb,
                broadcast_kwargs=dict(align_index=False),
            ),
            df4.vbt.combine(
                sr2,
                combine_func3_nb,
                broadcast_kwargs=dict(align_index=False),
            ),
        )

        target = pd.DataFrame(
            np.array([[232, 233, 234], [236, 237, 238], [240, 241, 242]]),
            index=pd.MultiIndex.from_tuples([("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]),
            columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func,
                10,
                b=100,
                concat=False,
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func_nb,
                10,
                100,
                jitted_loop=True,
                concat=False,
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            df4.vbt.combine(
                [10, sr2],
                combine_func,
                10,
                b=100,
                concat=False,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                target.values,
                index=pd.MultiIndex.from_tuples([("x6", "x2"), ("y6", "y2"), ("z6", "z2")], names=["i6", "i2"]),
                columns=target.columns,
            ),
        )
        target2 = pd.DataFrame(
            np.array([[121, 121, 121, 112, 113, 114], [122, 122, 122, 116, 117, 118], [123, 123, 123, 120, 121, 122]]),
            index=pd.MultiIndex.from_tuples([("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]),
            columns=pd.MultiIndex.from_tuples(
                [(0, "a6"), (0, "b6"), (0, "c6"), (1, "a6"), (1, "b6"), (1, "c6")],
                names=["combine_idx", "c6"],
            ),
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func,
                10,
                b=100,
                concat=True,
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func_nb,
                10,
                100,
                jitted_loop=True,
                concat=True,
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                lambda x, y, a, b=1: x + y + a + b,
                10,
                b=100,
                concat=True,
                keys=["a", "b"],
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                target2.values,
                index=target2.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a6"), ("a", "b6"), ("a", "c6"), ("b", "a6"), ("b", "b6"), ("b", "c6")],
                    names=[None, "c6"],
                ),
            ),
        )

        assert_frame_equal(
            sr2.vbt.combine(
                [10, 20],
                lambda x, y, a, b=1: x + y + a + b,
                vbt.Rep("y"),
                b=100,
                concat=True,
                keys=["a", "b"],
                broadcast_named_args=dict(y=df4),
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                np.array(
                    [[112, 113, 114, 122, 123, 124], [116, 117, 118, 126, 127, 128], [120, 121, 122, 130, 131, 132]],
                ),
                index=target2.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a6"), ("a", "b6"), ("a", "c6"), ("b", "a6"), ("b", "b6"), ("b", "c6")],
                    names=[None, "c6"],
                ),
            ),
        )
