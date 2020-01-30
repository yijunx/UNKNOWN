import yfinance as yf

def pull_stock(stock_name, start, end, save_path=None):
    df = yf.download(stock_name, start=start, end=end)
    if save_path:
        df.to_csv(save_path)
    return df
