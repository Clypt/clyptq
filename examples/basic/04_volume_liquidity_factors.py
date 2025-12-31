from datetime import timedelta

from clyptq import Constraints, CostModel, EngineMode
from clyptq.analytics.metrics import print_metrics
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.trading.engine import Engine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.factors.library.liquidity import AmihudFactor, EffectiveSpreadFactor
from clyptq.trading.factors.library.size import DollarVolumeSizeFactor
from clyptq.trading.factors.library.volume import VolumeRatioFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class VolumeStrategy(SimpleStrategy):
    def __init__(self):
        factors = [VolumeRatioFactor(short_window=5, long_window=20)]

        constraints = Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=5,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=5),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="Volume",
        )


class LiquidityStrategy(SimpleStrategy):
    def __init__(self):
        factors = [
            AmihudFactor(lookback=20),
            EffectiveSpreadFactor(lookback=20),
        ]

        constraints = Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=5,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=5),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="Liquidity",
        )


class SizeStrategy(SimpleStrategy):
    def __init__(self):
        factors = [DollarVolumeSizeFactor(lookback=20)]

        constraints = Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=5,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=5),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="Size",
        )


class CombinedStrategy(SimpleStrategy):
    def __init__(self):
        factors = [
            VolumeRatioFactor(short_window=5, long_window=20),
            AmihudFactor(lookback=20),
            DollarVolumeSizeFactor(lookback=20),
        ]

        constraints = Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=5,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=5),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="Combined",
        )


def run_strategy(strategy, store, start, end):
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    result = engine.run(start=start, end=end, verbose=False)
    return result


def main():
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "ADA/USDT",
        "XRP/USDT",
        "DOGE/USDT",
        "DOT/USDT",
        "MATIC/USDT",
        "AVAX/USDT",
    ]

    store = load_crypto_data(symbols=symbols, exchange="binance", timeframe="1d", days=180)

    date_range = store.get_date_range()
    start = date_range.end - timedelta(days=90)
    end = date_range.end

    print("=== Volume Strategy ===")
    volume_strategy = VolumeStrategy()
    volume_result = run_strategy(volume_strategy, store, start, end)
    print_metrics(volume_result.metrics)

    print("\n=== Liquidity Strategy ===")
    liquidity_strategy = LiquidityStrategy()
    liquidity_result = run_strategy(liquidity_strategy, store, start, end)
    print_metrics(liquidity_result.metrics)

    print("\n=== Size Strategy ===")
    size_strategy = SizeStrategy()
    size_result = run_strategy(size_strategy, store, start, end)
    print_metrics(size_result.metrics)

    print("\n=== Combined Strategy ===")
    combined_strategy = CombinedStrategy()
    combined_result = run_strategy(combined_strategy, store, start, end)
    print_metrics(combined_result.metrics)


if __name__ == "__main__":
    main()
