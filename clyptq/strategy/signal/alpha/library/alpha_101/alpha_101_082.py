"""Alpha 101_082: Open delta decay rank vs demeaned volume-open correlation decay rank min signal.

Formula: mul(min(rank(ts_decayed_linear(ts_delta({disk:open},1.46063),14.8717)),ts_rank(ts_decayed_linear(ts_corr(grouped_demean({disk:volume},{disk:industry_group_lv1}),add(mul({disk:open},0.634196),mul({disk:open},sub(1,0.634196))),17.4842),6.92131),13.4283)),-1)

Negative of min between open delta decay rank and demeaned volume-open correlation decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_082(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_082: Open delta decay rank vs demeaned volume-open correlation decay rank min.

    Negates the minimum of open delta decay rank and demeaned volume-open correlation decay ts_rank.
    """

    default_params = {
        "delta_window": 1,
        "decay_window1": 15,
        "corr_window": 17,
        "decay_window2": 7,
        "rank_window": 13,
    }

    @property
    def name(self) -> str:
        return "alpha_101_082"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_082."""
        open_ = data["open"]
        volume = data["volume"]

        # Part 1: Open delta decay rank
        # ts_delta(open, 1)
        open_delta = operator.ts_delta(open_, self.params["delta_window"])

        # ts_decayed_linear(open_delta, 15)
        open_decayed = operator.ts_decayed_linear(open_delta, self.params["decay_window1"])

        # rank(open_decayed)
        first_part = operator.rank(open_decayed)

        # Part 2: Demeaned volume-open correlation decay ts_rank
        # Demean volume (cross-sectional mean removal)
        volume_demeaned = operator.demean(volume)

        # Weighted open: open * 0.634196 + open * (1-0.634196) = open
        weighted_open = open_

        # ts_corr(volume_demeaned, weighted_open, 17)
        corr = operator.ts_corr(volume_demeaned, weighted_open, self.params["corr_window"])

        # ts_decayed_linear(corr, 7)
        corr_decayed = operator.ts_decayed_linear(corr, self.params["decay_window2"])

        # ts_rank(corr_decayed, 13)
        second_part = operator.ts_rank(corr_decayed, self.params["rank_window"])

        # min(first_part, second_part)
        min_result = operator.elem_min(first_part, second_part)

        # mul(min_result, -1)
        alpha = operator.mul(min_result, -1)

        return operator.ts_fillna(alpha, 0)
