import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_window = uic.loadUiType('./posterview.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Poster Viewer v.1.0')
        self.btn_squidgame.clicked.connect(self.btn_clicked_slot)
        self.btn_dp.clicked.connect(self.btn_clicked_slot)
        self.btn_kingdom1.clicked.connect(self.btn_clicked_slot)
        self.btn_kingdom2.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        self.lbl_squidgame.hide()
        self.lbl_dp.hide()
        self.lbl_kingdom1.hide()
        self.lbl_kingdom2.hide()
        btn = self.sender()
        if btn.objectName() == 'btn_squidgame':
            self.lbl_squidgame.show()
        elif btn.objectName() == 'btn_dp':
            self.lbl_dp.show()
        elif btn.objectName() == 'btn_kingdom1':
            self.lbl_kingdom1.show()
        elif btn.objectName() == 'btn_kingdom2':
            self.lbl_kingdom2.show()

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())