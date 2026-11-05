from lib import ui
from views import matchups

ui.header('Batter vs bowler', 'Who has the upper hand? Small samples are pulled toward the overall rate (empirical Bayes).')
matchups.render()
