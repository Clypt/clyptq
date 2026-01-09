"""Alpha 101_096: VWAP-volume rank correlation decay ts_rank vs close-amount ts_rank correlation argmax decay ts_rank max signal.

Formula: mul(max(ts_rank(ts_decayed_linear(ts_corr(rank({disk:vwap}),rank({disk:volume}),3.83878),4.16783),8.38151),ts_rank(ts_decayed_linear(ts_argmax(ts_corr(ts_rank({disk:close},7.45404),ts_rank(ts_mean({disk:amount},60),4.13242),3.65459),12.6556),14.0365),13.4143)),-1)

Negative of max between VWAP-volume rank correlation decay ts_rank and close-amount ts_rank correlation argmax decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_096(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_096: VWAP-volume rank correlation decay ts_rank vs close-amount ts_rank correlation argmax decay ts_rank max.

    Negates the maximum of two complex decay ts_rank values.
    """

    default_params = {
        "corr_window1": 4,
        "decay_window1": 4,
        "rank_window1": 8,
        "close_rank_window": 7,
        "amount_window": 60,
        "amount_rank_window": 4,
        "corr_window2": 4,
        "argmax_window": 13,
        "decay_window2": 14,
        "rank_window2": 13,
    }

    @property
    def name(self) -> str:
        return "alpha_101_096"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_096."""
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: VWAP-volume rank correlation decay ts_rank
        # rank(vwap)
        vwap_rank = operator.rank(vwap)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(vwap_rank, volume_rank, 4)
        vwap_vol_corr = operator.ts_corr(vwap_rank, volume_rank, self.params["corr_window1"])

        # ts_decayed_linear(vwap_vol_corr, 4)
        first_decayed = operator.ts_decayed_linear(vwap_vol_corr, self.params["decay_window1"])

        # ts_rank(first_decayed, 8)
        first_part = operator.ts_rank(first_decayed, self.params["rank_window1"])

        # Part 2: Close-amount ts_rank correlation argmax decay ts_rank
        # ts_rank(close, 7)
        close_tsrank = operator.ts_rank(close, self.params["close_rank_window"])

        # ts_mean(amount, 60)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_rank(amount_mean, 4)
        amount_tsrank = operator.ts_rank(amount_mean, self.params["amount_rank_window"])

        # ts_corr(close_tsrank, amount_tsrank, 4)
        close_amount_corr = operator.ts_corr(close_tsrank, amount_tsrank, self.params["corr_window2"])

        # ts_argmax(close_amount_corr, 13)
        corr_argmax = operator.ts_argmax(close_amount_corr, self.params["argmax_window"])

        # ts_decayed_linear(corr_argmax, 14)
        second_decayed = operator.ts_decayed_linear(corr_argmax, self.params["decay_window2"])

        # ts_rank(second_decayed, 13)
        second_part = operator.ts_rank(second_decayed, self.params["rank_window2"])

        # max(first_part, second_part)
        max_result = operator.elem_max(first_part, second_part)

        # mul(max_result, -1)
        alpha = operator.mul(max_result, -1)

        return operator.ts_fillna(alpha, 0)
