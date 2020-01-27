import sys
from Window_main import *

# 主函数
def main():
    app = QApplication(sys.argv)
    w = Window_main()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()