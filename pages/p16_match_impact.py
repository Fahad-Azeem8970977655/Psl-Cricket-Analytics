from lib import ui
from views import impact_match

ui.header('Match impact score', 'Batting + bowling + fielding + pressure + win bonus for every player in every match, in runs above an average player.')
impact_match.render()
