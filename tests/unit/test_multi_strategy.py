"""
Tests for multi-strategy framework (StrategyBlender).
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from clyptq.strategy.base import StrategyBlender, SimpleStrategy
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.factors.library.mean_reversion import ZScoreFactor
from clyptq.portfolio.construction import TopNConstructor, ScoreWeightedConstructor
from clyptq.types import Constraints
from clyptq.data.store import DataStore


def test_strategy_blender_init():
    """Test StrategyBlender initialization and validation."""
    momentum_strategy = SimpleStrategy(
        factors_list=[MomentumFactor(lookback=20)],
        constructor=TopNConstructor(top_n=5),
        constraints_obj=Constraints(
            max_position_size=0.2,
            min_position_size=0.05,
            max_gross_exposure=1.0,
            max_num_positions=10,
        ),
        name="Momentum",
    )

    mean_rev_strategy = SimpleStrategy(
        factors_list=[ZScoreFactor(lookback=20)],
        constructor=ScoreWeightedConstructor(),
        constraints_obj=Constraints(
            max_position_size=0.15,
            min_position_size=0.05,
            max_gross_exposure=0.8,
            max_num_positions=8,
        ),
        name="MeanRev",
    )

    blender = StrategyBlender(
        strategies={"Momentum": momentum_strategy, "MeanRev": mean_rev_strategy},
        allocations={"Momentum": 0.6, "MeanRev": 0.4},
    )

    assert len(blender.factors()) == 2
    assert blender.allocations == {"Momentum": 0.6, "MeanRev": 0.4}


def test_strategy_blender_allocations_validation():
    """Test allocation sum validation."""
    strategy = SimpleStrategy(
        factors_list=[MomentumFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        name="Test",
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        StrategyBlender(
            strategies={"A": strategy, "B": strategy},
            allocations={"A": 0.5, "B": 0.6},
        )


def test_strategy_blender_mismatched_keys():
    """Test mismatched strategies and allocations."""
    strategy = SimpleStrategy(
        factors_list=[MomentumFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        name="Test",
    )

    with pytest.raises(ValueError, match="same keys"):
        StrategyBlender(
            strategies={"A": strategy, "B": strategy},
            allocations={"A": 0.6, "C": 0.4},
        )


def test_strategy_blender_factor_tagging():
    """Test factor name tagging."""
    momentum_strategy = SimpleStrategy(
        factors_list=[MomentumFactor(lookback=20)],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        name="Momentum",
    )

    mean_rev_strategy = SimpleStrategy(
        factors_list=[ZScoreFactor(lookback=20)],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        name="MeanRev",
    )

    blender = StrategyBlender(
        strategies={"Momentum": momentum_strategy, "MeanRev": mean_rev_strategy},
        allocations={"Momentum": 0.6, "MeanRev": 0.4},
    )

    factor_names = [f.name for f in blender.factors()]
    assert "Momentum_Momentum" in factor_names
    assert "MeanRev_ZScore" in factor_names
    assert blender._factor_map["Momentum_Momentum"] == "Momentum"
    assert blender._factor_map["MeanRev_ZScore"] == "MeanRev"


def test_strategy_blender_constraints():
    """Test combined constraints (most restrictive)."""
    strategy_a = SimpleStrategy(
        factors_list=[MomentumFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(
            max_position_size=0.2,
            min_position_size=0.05,
            max_gross_exposure=1.0,
            max_num_positions=10,
        ),
        name="A",
    )

    strategy_b = SimpleStrategy(
        factors_list=[ZScoreFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(
            max_position_size=0.15,
            min_position_size=0.08,
            max_gross_exposure=0.8,
            max_num_positions=8,
        ),
        name="B",
    )

    blender = StrategyBlender(
        strategies={"A": strategy_a, "B": strategy_b},
        allocations={"A": 0.5, "B": 0.5},
    )

    constraints = blender.constraints()
    assert constraints.max_position_size == 0.15
    assert constraints.min_position_size == 0.08
    assert constraints.max_gross_exposure == 0.8
    assert constraints.max_num_positions == 8


def test_strategy_blender_schedule():
    """Test schedule selection (most frequent)."""
    strategy_daily = SimpleStrategy(
        factors_list=[MomentumFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        schedule_str="daily",
        name="Daily",
    )

    strategy_weekly = SimpleStrategy(
        factors_list=[ZScoreFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        schedule_str="weekly",
        name="Weekly",
    )

    blender = StrategyBlender(
        strategies={"Daily": strategy_daily, "Weekly": strategy_weekly},
        allocations={"Daily": 0.6, "Weekly": 0.4},
    )

    assert blender.schedule() == "daily"


def test_strategy_blender_warmup():
    """Test warmup periods (maximum)."""
    strategy_a = SimpleStrategy(
        factors_list=[MomentumFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        warmup=100,
        name="A",
    )

    strategy_b = SimpleStrategy(
        factors_list=[ZScoreFactor()],
        constructor=TopNConstructor(),
        constraints_obj=Constraints(),
        warmup=150,
        name="B",
    )

    blender = StrategyBlender(
        strategies={"A": strategy_a, "B": strategy_b},
        allocations={"A": 0.5, "B": 0.5},
    )

    assert blender.warmup_periods() == 150


def test_blended_constructor_weight_blending():
    """Test BlendedConstructor blends weights correctly."""
    from clyptq.portfolio.construction import BlendedConstructor

    momentum_strategy = SimpleStrategy(
        factors_list=[MomentumFactor(lookback=20)],
        constructor=TopNConstructor(top_n=3),
        constraints_obj=Constraints(
            max_position_size=0.5,
            min_position_size=0.01,
            max_gross_exposure=1.0,
            max_num_positions=10,
        ),
        name="Momentum",
    )

    mean_rev_strategy = SimpleStrategy(
        factors_list=[ZScoreFactor(lookback=20)],
        constructor=ScoreWeightedConstructor(),
        constraints_obj=Constraints(
            max_position_size=0.5,
            min_position_size=0.01,
            max_gross_exposure=1.0,
            max_num_positions=10,
        ),
        name="MeanRev",
    )

    factor_map = {"Momentum_Momentum": "Momentum", "MeanRev_ZScore": "MeanRev"}

    constructor = BlendedConstructor(
        strategies={"Momentum": momentum_strategy, "MeanRev": mean_rev_strategy},
        allocations={"Momentum": 0.6, "MeanRev": 0.4},
        factor_map=factor_map,
    )

    scores = {
        "BTC/USDT": 1.0,
        "ETH/USDT": 0.8,
        "SOL/USDT": 0.6,
        "ADA/USDT": 0.5,
    }

    constraints = Constraints(
        max_position_size=0.5,
        min_position_size=0.01,
        max_gross_exposure=1.0,
        max_num_positions=10,
    )

    weights = constructor.construct(scores, constraints)

    assert len(weights) > 0
    total_weight = sum(weights.values())
    assert total_weight <= 1.0
    assert all(w > 0 for w in weights.values())
    assert "BTC/USDT" in weights or "ETH/USDT" in weights


def test_blended_constructor_with_datastore():
    """Test BlendedConstructor with real DataStore."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    start = datetime(2024, 1, 1)

    store = DataStore()
    for symbol in symbols:
        dates = pd.date_range(start, periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.rand(100) * 1000,
            },
            index=dates,
        )
        store.add_ohlcv(symbol, df)

    momentum_strategy = SimpleStrategy(
        factors_list=[MomentumFactor(lookback=20)],
        constructor=TopNConstructor(top_n=2),
        constraints_obj=Constraints(
            max_position_size=0.5,
            min_position_size=0.1,
            max_gross_exposure=1.0,
            max_num_positions=5,
        ),
        name="Momentum",
    )

    mean_rev_strategy = SimpleStrategy(
        factors_list=[ZScoreFactor(lookback=20)],
        constructor=TopNConstructor(top_n=2),
        constraints_obj=Constraints(
            max_position_size=0.5,
            min_position_size=0.1,
            max_gross_exposure=1.0,
            max_num_positions=5,
        ),
        name="MeanRev",
    )

    blender = StrategyBlender(
        strategies={"Momentum": momentum_strategy, "MeanRev": mean_rev_strategy},
        allocations={"Momentum": 0.6, "MeanRev": 0.4},
    )

    timestamp = start + timedelta(days=50)
    data = store.get_view(timestamp)

    all_scores = {}
    for factor in blender.factors():
        try:
            scores = factor.compute(data)
            for symbol, score in scores.items():
                if symbol in all_scores:
                    all_scores[symbol] = (all_scores[symbol] + score) / 2
                else:
                    all_scores[symbol] = score
        except:
            continue

    if all_scores:
        constructor = blender.portfolio_constructor()
        constraints = blender.constraints()
        weights = constructor.construct(all_scores, constraints)

        assert isinstance(weights, dict)
        total_weight = sum(weights.values())
        assert total_weight <= constraints.max_gross_exposure
