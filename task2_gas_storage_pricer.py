"""
=================================================================
  Natural Gas Storage Contract Pricer
  Commodity Trading Desk  |  QR Team Prototype
  Task 2 — JP Morgan Quantitative Research Job Simulation
=================================================================

  Contract Value = Revenue from Sales
                 - Cost of Purchases
                 - Storage Costs
                 - Injection Costs
                 - Withdrawal Costs
                 - Transport Costs

  Key assumptions (per task specification):
    * No transport delay
    * Interest rates = zero
    * Market holidays / weekends / bank holidays not accounted for
    * Prices estimated via the Task 1 seasonal model if not provided

  This function generalises to multiple injection and withdrawal dates,
  allowing the desk to model complex multi-leg storage strategies.
=================================================================
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
from typing import List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PRICE ESTIMATOR (Task 1, embedded for self-contained pricing)
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
    df = df.sort_values('Dates').reset_index(drop=True)
    df['Prices'] = df['Prices'].astype(float)
    return df


def _date_to_t(date: pd.Timestamp, start: pd.Timestamp) -> float:
    whole_months = (date.year - start.year) * 12 + (date.month - start.month)
    fraction     = (date.day - 1) / 30
    return whole_months + fraction


def _seasonal_model(t, a, b, c, d):
    """Linear trend + single annual harmonic."""
    return a * t + b + c * np.sin(2 * np.pi * t / 12) + d * np.cos(2 * np.pi * t / 12)


def _fit_price_model(csv_path: str):
    """Returns (popt, start_date) for the fitted seasonal model."""
    df    = _load_data(csv_path)
    start = df['Dates'].min()
    t_vals = np.array([_date_to_t(d, start) for d in df['Dates']])
    popt, _ = curve_fit(_seasonal_model, t_vals, df['Prices'].values)
    return popt, start


def estimate_price(date_input, csv_path: str) -> float:
    """Return estimated gas price for a given date using the Task 1 model."""
    popt, start = _fit_price_model(csv_path)
    date  = pd.to_datetime(date_input)
    t     = _date_to_t(date, start)
    price = _seasonal_model(t, *popt)
    return round(float(price), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CONTRACT PRICING ENGINE  (Task 2)
# ─────────────────────────────────────────────────────────────────────────────

def price_storage_contract(
    injection_dates:             List[Union[str, datetime]],
    withdrawal_dates:            List[Union[str, datetime]],
    injection_volumes:           List[float],
    withdrawal_volumes:          List[float],
    max_volume:                  float,
    injection_rate:              float,
    withdrawal_rate:             float,
    monthly_storage_cost:        float,
    injection_cost_per_mmbtu:    float,
    withdrawal_cost_per_mmbtu:   float,
    transport_cost_per_leg:      float = 0.0,
    csv_path: str =              "data/nat_gas_prices.csv",
    injection_prices:            Optional[List[float]] = None,
    withdrawal_prices:           Optional[List[float]] = None,
    verbose:                     bool  = True,
) -> dict:
    """
    Price a natural gas storage contract.

    The contract value represents the fair price at which both the trading
    desk and the client would be willing to enter the agreement:

        Value = Σ(withdrawal revenue) - Σ(injection cost)
                - storage cost - injection fees - withdrawal fees
                - transport costs

    Parameters
    ----------
    injection_dates : list of dates when gas is purchased and injected
    withdrawal_dates : list of dates when gas is withdrawn and sold
    injection_volumes : MMBtu of gas injected on each injection date
                        (must match length of injection_dates)
    withdrawal_volumes : MMBtu of gas withdrawn on each withdrawal date
                         (must match length of withdrawal_dates)
    max_volume : maximum MMBtu the storage facility can hold at any time
    injection_rate : maximum MMBtu that can be injected per day
    withdrawal_rate : maximum MMBtu that can be withdrawn per day
    monthly_storage_cost : fixed fee paid to the storage facility per month
                           the gas is stored ($)
    injection_cost_per_mmbtu : variable cost per MMBtu injected ($)
    withdrawal_cost_per_mmbtu : variable cost per MMBtu withdrawn ($)
    transport_cost_per_leg : one-way transport cost per injection/withdrawal ($)
                             applied once per injection and once per withdrawal
                             (default 0 — no transport cost)
    csv_path : path to the monthly natural gas price CSV (Task 1 data)
    injection_prices : optional — override model prices at injection dates ($)
    withdrawal_prices : optional — override model prices at withdrawal dates ($)
    verbose : print a detailed cash flow breakdown (default True)

    Returns
    -------
    dict with keys:
        contract_value       : net value of the contract ($)
        total_revenue        : gross revenue from gas sales ($)
        total_purchase_cost  : gross cost of gas purchases ($)
        total_storage_cost   : total periodic storage fees ($)
        total_injection_fees : total variable injection costs ($)
        total_withdrawal_fees: total variable withdrawal costs ($)
        total_transport_cost : total transport costs ($)
        legs                 : list of per-leg cash flow dicts
        warnings             : list of constraint violation messages

    Raises
    ------
    ValueError : if volumes are negative, dates are inconsistent, or
                 more gas is withdrawn than was ever injected.
    """

    # ── Input validation ──────────────────────────────────────────────────
    if len(injection_dates) != len(injection_volumes):
        raise ValueError("injection_dates and injection_volumes must have the same length.")
    if len(withdrawal_dates) != len(withdrawal_volumes):
        raise ValueError("withdrawal_dates and withdrawal_volumes must have the same length.")
    if any(v < 0 for v in injection_volumes + withdrawal_volumes):
        raise ValueError("Volumes cannot be negative.")
    if monthly_storage_cost < 0:
        raise ValueError("Storage cost cannot be negative.")

    injection_dates_pd  = [pd.to_datetime(d) for d in injection_dates]
    withdrawal_dates_pd = [pd.to_datetime(d) for d in withdrawal_dates]

    # ── Price estimation ──────────────────────────────────────────────────
    if injection_prices is None:
        injection_prices = [estimate_price(d, csv_path) for d in injection_dates_pd]
    if withdrawal_prices is None:
        withdrawal_prices = [estimate_price(d, csv_path) for d in withdrawal_dates_pd]

    # ── Constraint checks ─────────────────────────────────────────────────
    contract_warnings = []

    total_injected   = sum(injection_volumes)
    total_withdrawn  = sum(withdrawal_volumes)

    if total_withdrawn > total_injected:
        raise ValueError(
            f"Cannot withdraw {total_withdrawn:,.0f} MMBtu: "
            f"only {total_injected:,.0f} MMBtu was ever injected."
        )

    # Check injection rate constraints
    for i, (date, vol) in enumerate(zip(injection_dates_pd, injection_volumes)):
        if vol > injection_rate:
            contract_warnings.append(
                f"Injection leg {i+1} ({date.date()}): volume {vol:,.0f} MMBtu "
                f"exceeds daily injection rate {injection_rate:,.0f} MMBtu/day. "
                f"Real injection would span multiple days."
            )

    # Check withdrawal rate constraints
    for i, (date, vol) in enumerate(zip(withdrawal_dates_pd, withdrawal_volumes)):
        if vol > withdrawal_rate:
            contract_warnings.append(
                f"Withdrawal leg {i+1} ({date.date()}): volume {vol:,.0f} MMBtu "
                f"exceeds daily withdrawal rate {withdrawal_rate:,.0f} MMBtu/day."
            )

    # Check max_volume constraint (running inventory)
    all_events = (
        [(d, +v, 'injection')  for d, v in zip(injection_dates_pd,  injection_volumes)] +
        [(d, -v, 'withdrawal') for d, v in zip(withdrawal_dates_pd, withdrawal_volumes)]
    )
    all_events.sort(key=lambda x: x[0])

    running_inventory = 0.0
    for event_date, delta_vol, event_type in all_events:
        running_inventory += delta_vol
        if running_inventory > max_volume:
            contract_warnings.append(
                f"Storage capacity exceeded on {event_date.date()}: "
                f"inventory would reach {running_inventory:,.0f} MMBtu "
                f"vs max {max_volume:,.0f} MMBtu."
            )
        if running_inventory < 0:
            contract_warnings.append(
                f"Inventory goes negative on {event_date.date()} "
                f"({running_inventory:,.0f} MMBtu). "
                f"Withdrawal scheduled before sufficient injection."
            )

    # ── Storage duration ──────────────────────────────────────────────────
    # Storage cost = monthly_storage_cost x number of months between
    # first injection and last withdrawal.
    all_injection_dates   = sorted(injection_dates_pd)
    all_withdrawal_dates  = sorted(withdrawal_dates_pd)

    storage_start = min(all_injection_dates)
    storage_end   = max(all_withdrawal_dates)

    storage_months = (
        (storage_end.year  - storage_start.year) * 12 +
        (storage_end.month - storage_start.month)
    )
    # At minimum, charge for at least one month if any gas is stored
    storage_months = max(storage_months, 1) if total_injected > 0 else 0
    total_storage_cost = monthly_storage_cost * storage_months

    # ── Cash flows ────────────────────────────────────────────────────────
    legs = []

    # Injection legs — cash outflows
    total_purchase_cost  = 0.0
    total_injection_fees = 0.0
    total_transport_in   = 0.0

    for i, (date, vol, price) in enumerate(
            zip(injection_dates_pd, injection_volumes, injection_prices)):
        purchase_cost  = vol * price
        injection_fee  = vol * injection_cost_per_mmbtu
        transport_cost = transport_cost_per_leg

        total_purchase_cost  += purchase_cost
        total_injection_fees += injection_fee
        total_transport_in   += transport_cost

        legs.append({
            'leg':           f"Injection {i+1}",
            'date':          date.date(),
            'volume_mmbtu':  vol,
            'price_per_mmbtu': price,
            'purchase_cost': -purchase_cost,
            'variable_fee':  -injection_fee,
            'transport':     -transport_cost,
            'net_cashflow':  -(purchase_cost + injection_fee + transport_cost),
        })

    # Withdrawal legs — cash inflows
    total_revenue         = 0.0
    total_withdrawal_fees = 0.0
    total_transport_out   = 0.0

    for i, (date, vol, price) in enumerate(
            zip(withdrawal_dates_pd, withdrawal_volumes, withdrawal_prices)):
        revenue        = vol * price
        withdrawal_fee = vol * withdrawal_cost_per_mmbtu
        transport_cost = transport_cost_per_leg

        total_revenue         += revenue
        total_withdrawal_fees += withdrawal_fee
        total_transport_out   += transport_cost

        legs.append({
            'leg':             f"Withdrawal {i+1}",
            'date':            date.date(),
            'volume_mmbtu':    vol,
            'price_per_mmbtu': price,
            'sale_revenue':    +revenue,
            'variable_fee':    -withdrawal_fee,
            'transport':       -transport_cost,
            'net_cashflow':    revenue - withdrawal_fee - transport_cost,
        })

    total_transport_cost = total_transport_in + total_transport_out

    contract_value = (
        total_revenue
        - total_purchase_cost
        - total_storage_cost
        - total_injection_fees
        - total_withdrawal_fees
        - total_transport_cost
    )

    # ── Verbose output ────────────────────────────────────────────────────
    if verbose:
        print("=" * 65)
        print("  NATURAL GAS STORAGE CONTRACT — PRICING SUMMARY")
        print("=" * 65)
        print(f"\n  Storage period : {storage_start.date()}  →  {storage_end.date()}")
        print(f"  Storage months : {storage_months}")
        print(f"  Max capacity   : {max_volume:>12,.0f} MMBtu")
        print(f"  Total injected : {total_injected:>12,.0f} MMBtu")
        print(f"  Total withdrawn: {total_withdrawn:>12,.0f} MMBtu")

        if contract_warnings:
            print(f"\n  ⚠  CONSTRAINT WARNINGS ({len(contract_warnings)})")
            for w in contract_warnings:
                print(f"     • {w}")

        print(f"\n  {'─'*63}")
        print(f"  INJECTION LEGS")
        print(f"  {'─'*63}")
        for leg in [l for l in legs if 'Injection' in l['leg']]:
            print(f"  {leg['leg']:<14}  {str(leg['date']):<12}  "
                  f"{leg['volume_mmbtu']:>10,.0f} MMBtu  "
                  f"@ ${leg['price_per_mmbtu']:.4f}/MMBtu  "
                  f"→  Net: ${leg['net_cashflow']:>12,.2f}")

        print(f"\n  {'─'*63}")
        print(f"  WITHDRAWAL LEGS")
        print(f"  {'─'*63}")
        for leg in [l for l in legs if 'Withdrawal' in l['leg']]:
            print(f"  {leg['leg']:<14}  {str(leg['date']):<12}  "
                  f"{leg['volume_mmbtu']:>10,.0f} MMBtu  "
                  f"@ ${leg['price_per_mmbtu']:.4f}/MMBtu  "
                  f"→  Net: ${leg['net_cashflow']:>12,.2f}")

        print(f"\n  {'─'*63}")
        print(f"  CASH FLOW BREAKDOWN")
        print(f"  {'─'*63}")
        print(f"  {'Revenue from sales':<35}  ${total_revenue:>14,.2f}")
        print(f"  {'Purchase cost of gas':<35}  ${-total_purchase_cost:>14,.2f}")
        print(f"  {'Storage fees':<35}  ${-total_storage_cost:>14,.2f}")
        print(f"  {'Injection fees':<35}  ${-total_injection_fees:>14,.2f}")
        print(f"  {'Withdrawal fees':<35}  ${-total_withdrawal_fees:>14,.2f}")
        print(f"  {'Transport costs':<35}  ${-total_transport_cost:>14,.2f}")
        print(f"  {'─'*53}")
        sign   = "+" if contract_value >= 0 else ""
        status = "PROFITABLE" if contract_value > 0 else ("BREAK-EVEN" if contract_value == 0 else "LOSS-MAKING")
        print(f"  {'CONTRACT VALUE':<35}  ${sign}{contract_value:>13,.2f}   [{status}]")
        print("=" * 65)

    return {
        'contract_value':        round(contract_value,        2),
        'total_revenue':         round(total_revenue,         2),
        'total_purchase_cost':   round(total_purchase_cost,   2),
        'total_storage_cost':    round(total_storage_cost,    2),
        'total_injection_fees':  round(total_injection_fees,  2),
        'total_withdrawal_fees': round(total_withdrawal_fees, 2),
        'total_transport_cost':  round(total_transport_cost,  2),
        'storage_months':        storage_months,
        'legs':                  legs,
        'warnings':              contract_warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TEST CASES  (per task specification)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    CSV = "data/nat_gas_prices.csv"

    print("\n" + "=" * 65)
    print("  TEST CASE 1 — Simple Summer Buy / Winter Sell")
    print("  (From task example: buy summer, sell winter)")
    print("=" * 65)
    result1 = price_storage_contract(
        injection_dates          = ["2022-06-30"],
        withdrawal_dates         = ["2022-12-31"],
        injection_volumes        = [1_000_000],       # 1 million MMBtu
        withdrawal_volumes       = [1_000_000],
        max_volume               = 1_500_000,
        injection_rate           = 50_000,            # 50k MMBtu/day max
        withdrawal_rate          = 50_000,
        monthly_storage_cost     = 100_000,           # $100K/month (task example)
        injection_cost_per_mmbtu = 0.01,              # $10K per 1M MMBtu
        withdrawal_cost_per_mmbtu= 0.01,
        transport_cost_per_leg   = 50_000,            # $50K each way (task example)
        csv_path                 = CSV,
        verbose                  = True,
    )

    print("\n" + "=" * 65)
    print("  TEST CASE 2 — Multi-leg Strategy (2 injections, 2 withdrawals)")
    print("=" * 65)
    result2 = price_storage_contract(
        injection_dates          = ["2022-04-30", "2022-06-30"],
        withdrawal_dates         = ["2022-11-30", "2023-01-31"],
        injection_volumes        = [500_000, 500_000],
        withdrawal_volumes       = [600_000, 400_000],
        max_volume               = 1_200_000,
        injection_rate           = 100_000,
        withdrawal_rate          = 100_000,
        monthly_storage_cost     = 80_000,
        injection_cost_per_mmbtu = 0.005,
        withdrawal_cost_per_mmbtu= 0.005,
        transport_cost_per_leg   = 25_000,
        csv_path                 = CSV,
        verbose                  = True,
    )

    print("\n" + "=" * 65)
    print("  TEST CASE 3 — Break-even Check (manual price override)")
    print("  Injection and withdrawal at same price → should be loss")
    print("=" * 65)
    result3 = price_storage_contract(
        injection_dates          = ["2023-03-31"],
        withdrawal_dates         = ["2023-06-30"],
        injection_volumes        = [750_000],
        withdrawal_volumes       = [750_000],
        max_volume               = 1_000_000,
        injection_rate           = 75_000,
        withdrawal_rate          = 75_000,
        monthly_storage_cost     = 50_000,
        injection_cost_per_mmbtu = 0.01,
        withdrawal_cost_per_mmbtu= 0.01,
        transport_cost_per_leg   = 0,
        csv_path                 = CSV,
        injection_prices         = [12.00],   # same price both legs
        withdrawal_prices        = [12.00],   # forces a loss due to costs
        verbose                  = True,
    )

    print("\n" + "=" * 65)
    print("  TEST CASE 4 — Future Contract (extrapolation, 2025)")
    print("=" * 65)
    result4 = price_storage_contract(
        injection_dates          = ["2025-04-30"],
        withdrawal_dates         = ["2025-10-31"],
        injection_volumes        = [800_000],
        withdrawal_volumes       = [800_000],
        max_volume               = 1_000_000,
        injection_rate           = 60_000,
        withdrawal_rate          = 60_000,
        monthly_storage_cost     = 90_000,
        injection_cost_per_mmbtu = 0.008,
        withdrawal_cost_per_mmbtu= 0.008,
        transport_cost_per_leg   = 30_000,
        csv_path                 = CSV,
        verbose                  = True,
    )

    # Summary table
    print("\n" + "=" * 65)
    print("  ALL TEST CASES — CONTRACT VALUE SUMMARY")
    print("=" * 65)
    cases = [
        ("TC1: Simple buy/sell",       result1),
        ("TC2: Multi-leg strategy",    result2),
        ("TC3: Break-even check",      result3),
        ("TC4: Future contract 2025",  result4),
    ]
    for label, r in cases:
        val    = r['contract_value']
        status = "✓ PROFIT" if val > 0 else ("✗ LOSS  " if val < 0 else "= BREAK-EVEN")
        print(f"  {label:<30}  ${val:>14,.2f}   {status}")
    print("=" * 65)