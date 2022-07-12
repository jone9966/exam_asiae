import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_window = uic.loadUiType('./calculator.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Calculator Ver.1.0.0')

        self.first_input_flag = True
        self.first_number = 0
        self.opcode = ''
        btn_number_list = [self.btn_0, self.btn_1, self.btn_2, self.btn_3, self.btn_4,
                           self.btn_5, self.btn_6, self.btn_7, self.btn_8, self.btn_9]
        btn_opcode_list = [self.btn_plus, self.btn_minus, self.btn_mul, self.btn_div, self.btn_equal]
        shortcut_opcode = ['+', '-', '*', '/', 'Return']
        for index, btn in enumerate(btn_number_list):
            btn.clicked.connect(self.btn_number_clicked_slot)
            btn.setShortcut(str(index))
        for index, btn in enumerate(btn_opcode_list):
            btn.clicked.connect(self.btn_opcode_clicked_slot)
            btn.setShortcut(shortcut_opcode[index])
        self.btn_clear.clicked.connect(self.btn_clear_clicked_slot)

    def btn_number_clicked_slot(self):
        btn = self.sender()
        if self.first_input_flag :
            self.first_input_flag = False
            self.lbl_result.setText('')
        if self.lbl_result.text() == '0':
            self.lbl_result.setText('')
        self.lbl_result.setText(self.lbl_result.text() + btn.objectName()[4:])

    def btn_opcode_clicked_slot(self):
        if not self.first_input_flag:
            if self.opcode !='equal':
                self.calculate()
            self.first_number = float(self.lbl_result.text())
            self.first_input_flag = True
        self.opcode = self.sender().objectName()[4:]


    def calculate(self):
        second_number = float(self.lbl_result.text())
        if self.opcode == 'plus':
            self.lbl_result.setText(str(self.first_number + second_number))
        elif self.opcode == 'minus':
            self.lbl_result.setText(str(self.first_number - second_number))
        elif self.opcode == 'mul':
            self.lbl_result.setText(str(self.first_number * second_number))
        elif self.opcode == 'div':
            if second_number :
                self.lbl_result.setText(str(self.first_number / second_number))
            else : self.lbl_result.setText('infinity')

    def btn_clear_clicked_slot(self):
        self.first_input_flag = True
        self.lbl_result.setText('0')
        self.opcode = ''
        self.first_number = None


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())