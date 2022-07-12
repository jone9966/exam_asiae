import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_window = uic.loadUiType('./signal_slot.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_1.clicked.connect(self.btn_clicked_slot)
        self.lcdNumber.setVisible(False)

    def btn_clicked_slot(self):
        self.lbl_1.setText("Hello World!")

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())