from lib import ui
from views import team_strength

ui.header('Team strength (Elo)', 'A rating that moves after every match. Higher = stronger recent results.')
team_strength.render()
