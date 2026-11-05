from lib import ui
from views import batting_order, partnerships

ui.header('Partnerships & batting order', 'Who bats together, where batters come in, and what each position delivers.')
ui.views({'Partnerships': partnerships.render, 'Batting order': batting_order.render}, key='view_po')
