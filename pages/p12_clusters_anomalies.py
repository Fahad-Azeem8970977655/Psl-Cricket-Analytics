from lib import ui
from views import anomalies, clusters

ui.header('Clusters, similar players & anomalies', 'Unsupervised ML: playing-style groups, nearest neighbours and statistical oddities.')
ui.views({'Clusters & similar players': clusters.render, 'Anomalies': anomalies.render}, key='view_ca')
