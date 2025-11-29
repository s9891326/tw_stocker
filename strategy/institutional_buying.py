from typing import List, Tuple

import numpy as np
import pandas as pd


def trade(
    df: pd.DataFrame, continuous_days: int = 3, **kwargs
) -> Tuple[List[int], List[int], List[bool], List[bool], float, float]:
    """
    Institutional Continuous Buying Strategy

    Buy Signal:
        - Foreign Investors (foreign_net_buy) > 0 AND Investment Trusts (invest_trust_net_buy) > 0
        - Condition must hold for `continuous_days` consecutive days (including today).

    Sell Signal:
        - If currently holding position:
        - Foreign Investors < 0 OR Investment Trusts < 0

    Returns:
        states_buy, states_sell, states_entry, states_exit, total_gains, invest
    """

    # Ensure required columns exist
    required_cols = ["foreign_net_buy", "invest_trust_net_buy", "close"]
    for col in required_cols:
        if col not in df.columns:
            # If data is missing, return empty results
            n = len(df)
            return [], [], [False] * n, [False] * n, 0.0, 0.0

    # Fill NaNs with 0 for institutional data to avoid breaking logic
    df_clean = df.copy()
    df_clean["foreign_net_buy"] = df_clean["foreign_net_buy"].fillna(0)
    df_clean["invest_trust_net_buy"] = df_clean["invest_trust_net_buy"].fillna(0)

    close_prices = df_clean["close"].values
    foreign = df_clean["foreign_net_buy"].values
    trust = df_clean["invest_trust_net_buy"].values

    n = len(df)
    states_buy = []
    states_sell = []
    states_entry = [False] * n
    states_exit = [False] * n

    current_money = 10000.0  # Initial capital for simulation
    initial_money = current_money
    shares = 0

    # Simulation parameters (simplified)
    fee_rate = kwargs.get("fee_rate", 0.001425)
    tax_rate = kwargs.get("tax_rate", 0.003)

    holding = False

    for i in range(continuous_days - 1, n):
        # Check Buy Condition
        # Both must be > 0 for the last `continuous_days`
        is_buy_signal = True
        for k in range(continuous_days):
            idx = i - k
            if not (foreign[idx] > 0 and trust[idx] > 0):
                is_buy_signal = False
                break

        # Check Sell Condition (only if holding)
        is_sell_signal = False
        if holding:
            if foreign[i] < 0 or trust[i] < 0:
                is_sell_signal = True

        price = close_prices[i]

        if holding:
            if is_sell_signal:
                revenue = shares * price
                cost = revenue * (fee_rate + tax_rate)
                current_money += revenue - cost
                shares = 0
                holding = False
                states_sell.append(i)
                states_exit[i] = True
            else:
                states_entry[i] = True
        else:
            if is_buy_signal:
                cost_per_share = price * (1 + fee_rate)
                if current_money > cost_per_share:
                    buy_shares = int(current_money // cost_per_share)
                    if buy_shares > 0:
                        cost = buy_shares * price * (1 + fee_rate)
                        current_money -= cost
                        shares += buy_shares
                        holding = True
                        states_buy.append(i)
                        states_entry[i] = True

    # Calculate final value
    final_assets = current_money + (shares * close_prices[-1] if shares > 0 else 0)
    total_gains = final_assets - initial_money
    invest_return = (total_gains / initial_money) * 100 if initial_money > 0 else 0

    return (
        states_buy,
        states_sell,
        states_entry,
        states_exit,
        total_gains,
        invest_return,
    )
