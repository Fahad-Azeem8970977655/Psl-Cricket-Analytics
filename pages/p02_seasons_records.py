from lib import ui
from views import records, seasons

ui.header('Seasons, champions & records', "Finals, league tables, awards and the league's extremes.")
ui.views({'Seasons & champions': seasons.render, 'Records & extremes': records.render}, key='view_sr')
