from lib import ui
from views import phases, venues

ui.header('Venues & phases', 'Scoring by ground, and how an innings is paced.')
ui.views({'Venues': venues.render, 'Phases & overs': phases.render}, key='view_vp')
