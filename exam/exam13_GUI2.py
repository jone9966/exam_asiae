import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

class Exam(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lbl = QLabel('Hello world!', self)
        self.lbl.move(20, 30)
        self.btn = QPushButton('Button', self)
        self.btn.move(20, 60)
        self.setGeometry(300, 300, 300, 500)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())