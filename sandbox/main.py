import sys
from PyQt5 import QtWidgets

from sandbox.diffeq.double_pendulum import DoublePendulum

app = QtWidgets.QApplication(sys.argv)
window = DoublePendulum()
window.show()
app.exec_()
