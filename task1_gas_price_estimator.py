from datetime import datetime

prices = {
"2020-10-31": 10.1,
"2020-11-30": 10.3,
# continue same format...
}

def price_gas_contract(
    injection_dates,
    withdrawal_dates,
    prices,
    volume,
    max_storage,
    injection_rate,
    withdrawal_rate,
    storage_cost_per_day
):
    total_value = 0
    current_storage = 0

    for inj_date, wd_date in zip(injection_dates, withdrawal_dates):

        if inj_date not in prices or wd_date not in prices:
            raise ValueError("Price missing for selected dates")

        inj = datetime.strptime(inj_date, "%Y-%m-%d")
        wd = datetime.strptime(wd_date, "%Y-%m-%d")

        days = (wd - inj).days

        injected = min(injection_rate, volume, max_storage - current_storage)
        withdrawn = min(withdrawal_rate, injected)

        buy_price = prices[inj_date]
        sell_price = prices[wd_date]

        buy_cost = injected * buy_price
        sell_value = withdrawn * sell_price
        storage_cost = injected * storage_cost_per_day * days

        total_value += (sell_value - buy_cost - storage_cost)

        current_storage += injected - withdrawn

    return total_value