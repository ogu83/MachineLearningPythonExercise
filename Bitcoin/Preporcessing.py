#from exchanges.bitfinex import Bitfinex
from exchanges.coindesk import CoinDesk

curr_price = CoinDesk().get_current_price()
print(curr_price)

#print(Bitfinex().get_current_price(currency='EUR'))
#print(Bitfinex().get_current_price(currency='TRY'))