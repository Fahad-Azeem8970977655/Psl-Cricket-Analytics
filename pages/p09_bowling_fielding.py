from lib import ui
from views import bowler, fielding

ui.header('Bowling & fielding', 'Bowler profiles, and catches, run-outs and stumpings.')
ui.views({'Bowler profile': bowler.render, 'Fielding': fielding.render}, key='view_bf')
