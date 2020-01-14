
from pytrends.request import TrendReq


pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["blockchain"]
pytrends.build_payload(kw_list, cat=0, timeframe='today 2-y', geo='', gprop='')

inte_over_time = pytrends.interest_over_time()
print(inte_over_time.tail())
