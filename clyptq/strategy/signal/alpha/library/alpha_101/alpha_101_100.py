"""Alpha 101_100: Complex multi-demeaned price position-volume and correlation factor signal.

Formula: sub(0,mul(1,mul(sub(mul(1.5,twise_a_scale(grouped_demean(grouped_demean(rank(mul(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:high},{disk:low})),{disk:volume})),{disk:industry_group_lv3}),{disk:industry_group_lv3}))),twise_a_scale(grouped_demean(sub(ts_corr({disk:close},rank(ts_mean({disk:amount},20)),5),rank(ts_argmin({disk:close},30))),{disk:industry_group_lv3}))),div({disk:volume},ts_mean({disk:amount},20)))))

Negative of complex multi-demeaned price position-volume and correlation factor.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_100(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_100: Complex multi-demeaned price position-volume and correlation factor.

    Negates a complex factor combining multi-demeaned price position-volume with correlation-argmin difference.
    """

    default_params = {
        "amount_window": 20,
        "corr_window": 5,
        "argmin_window": 30,
        "scale_factor": 1.5,
    }

    @property
    def name(self) -> str:
        return "alpha_101_100"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_100."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Multi-demeaned price position-volume
        # Williams %R like calculation
        close_low = operator.sub(close, low)
        high_close = operator.sub(high, close)
        high_low = operator.sub(high, low)

        williams_r_num = operator.sub(close_low, high_close)
        williams_r = operator.div(williams_r_num, high_low)

        # mul(williams_r, volume)
        wr_volume = operator.mul(williams_r, volume)

        # rank(wr_volume)
        wr_rank = operator.rank(wr_volume)

        # Double demeaning (cross-sectional mean removal as proxy for industry demeaning)
        first_demean = operator.demean(wr_rank)
        second_demean = operator.demean(first_demean)

        # twise_a_scale(second_demean) and mul(1.5, ...)
        first_scaled = operator.twise_a_scale(second_demean, 1)
        first_part = operator.mul(self.params["scale_factor"], first_scaled)

        # Part 2: Correlation and argmin difference
        # ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # rank(amount_mean)
        amount_rank = operator.rank(amount_mean)

        # ts_corr(close, amount_rank, 5)
        close_amount_corr = operator.ts_corr(close, amount_rank, self.params["corr_window"])

        # ts_argmin(close, 30)
        close_argmin = operator.ts_argmin(close, self.params["argmin_window"])

        # rank(close_argmin)
        argmin_rank = operator.rank(close_argmin)

        # sub(close_amount_corr, argmin_rank)
        corr_diff = operator.sub(close_amount_corr, argmin_rank)

        # Demean (cross-sectional mean removal as proxy for industry demeaning)
        corr_demeaned = operator.demean(corr_diff)

        # twise_a_scale(corr_demeaned)
        second_part = operator.twise_a_scale(corr_demeaned, 1)

        # Part 3: Volume ratio
        # div(volume, amount_mean)
        volume_ratio = operator.div(volume, amount_mean)

        # Final calculation
        # sub(first_part, second_part)
        main_diff = operator.sub(first_part, second_part)

        # mul(main_diff, volume_ratio)
        product = operator.mul(main_diff, volume_ratio)

        # Negate
        alpha = operator.mul(product, -1)

        return operator.ts_fillna(alpha, 0)
