from lib import ui
from views import predictor, winprob

ui.header('Win prediction', 'Before the first ball (honest results) and during the match (ball-by-ball models).')
ui.views({'Before the match': predictor.render, 'During the match': winprob.render}, key='view_wp')
