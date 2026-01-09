"""Alpha 101_071: Close-amount correlation decay rank vs price difference decay rank max signal.

Formula: max(ts_rank(ts_decayed_linear(ts_corr(ts_rank({disk:close},3.43976),ts_rank(ts_mean({disk:amount},180),12.0647),18.0175),4.20501),15.6948),ts_rank(ts_decayed_linear(pow(rank(sub(add({disk:low},{disk:open}),add({disk:vwap},{disk:vwap}))),2),16.4662),4.4388))

Maximum of close-amount correlation decay ts_rank and price difference squared decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_071(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_071: Close-amount correlation decay rank vs price difference decay rank max.

    Returns the maximum of two complex decay ts_rank calculations.
    """

    default_params = {
        "close_rank_window": 3,
        "amount_window": 180,
        "amount_rank_window": 12,
        "corr_window": 18,
        "decay_window1": 4,
        "corr_final_window": 16,
        "decay_window2": 16,
        "price_final_window": 4,
    }

    @property
    def name(self) -> str:
        return "alpha_101_071"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_071."""
        open_ = data["open"]
        low = data["low"]
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

        # Part 1: Close-amount correlation decay ts_rank
        # ts_rank(close, 3)
        close_tsrank = operator.ts_rank(close, self.params["close_rank_window"])

        # ts_mean(amount, 180)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_rank(amount_mean, 12)
        amount_tsrank = operator.ts_rank(amount_mean, self.params["amount_rank_window"])

        # ts_corr(close_tsrank, amount_tsrank, 18)
        corr = operator.ts_corr(close_tsrank, amount_tsrank, self.params["corr_window"])

        # ts_decayed_linear(corr, 4)
        corr_decayed = operator.ts_decayed_linear(corr, self.params["decay_window1"])

        # ts_rank(corr_decayed, 16)
        first_part = operator.ts_rank(corr_decayed, self.params["corr_final_window"])

        # Part 2: Price difference squared decay ts_rank
        # low + open
        low_open_sum = operator.add(low, open_)

        # 2 * vwap
        vwap_double = operator.add(vwap, vwap)

        # (low + open) - 2*vwap
        price_diff = operator.sub(low_open_sum, vwap_double)

        # rank(price_diff)
        price_rank = operator.rank(price_diff)

        # pow(price_rank, 2)
        price_squared = operator.pow(price_rank, 2)

        # ts_decayed_linear(price_squared, 16)
        price_decayed = operator.ts_decayed_linear(price_squared, self.params["decay_window2"])

        # ts_rank(price_decayed, 4)
        second_part = operator.ts_rank(price_decayed, self.params["price_final_window"])

        # max(first_part, second_part)
        alpha = operator.elem_max(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
