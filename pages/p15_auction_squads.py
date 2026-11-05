from lib import ui
from views import auction_squads, value_money

ui.header('Auction, squads & value for money', 'Who was bought, what they cost, who leads each team, and what 2026 prices bought.')
ui.views({'Auction, squads & management': auction_squads.render, 'Value for money (2026)': value_money.render}, key='view_as')
