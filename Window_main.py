from Ui_main import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import copy

from Train import *

class Window_main(QMainWindow, Ui_main):
    c = 1
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.widget.setLayout(layout)

        self.maze = None
        self.timer = QtCore.QTimer(self)
        self.Train = Train()

        self.setSlot()
        self.show()

    # 设置槽函数
    def setSlot(self):
        self.pushButton_generateMaze.clicked.connect(self.generateMaze)
        self.pushButton_solveMaze.clicked.connect(self.solveMaze)
        self.Train.epochSignal.connect(self.setEpoch)
        self.Train.lossSignal.connect(self.setLoss)
        self.Train.winRateSignal.connect(self.setWinRate)
        self.Train.timeSignal.connect(self.setTime)

    def generateMaze(self):
        self.mazeSize = int(self.comboBox_mazeSize.currentText())
        self.maze = generate_maze(size=self.mazeSize)
        self.qmaze = Qmaze(self.maze)
        self.ax = self.fig.subplots()
        self.showCurCanvas()


    def solveMaze(self):
        if not np.any(self.maze):
            QMessageBox.warning(self, "Warning", "请先生成迷宫！")
            return False
        self.showCurCanvas()
        self.model = build_model(self.maze)
        self.Train.qtrain(self.model, self.maze, n_epochs=100, max_memory=8*self.maze.size, data_size=32)

        self.qmaze.reset(self.qmaze.free_cells[0])
        envstate = self.qmaze.observe()
        counts = 0
        self.mazeHistory = Queue()
        while True:
            self.ax.cla()
            prev_envstate = envstate
            # get next action
            q = self.model.predict(prev_envstate)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            envstate, reward, game_status = self.qmaze.act(action)
            self.mazeHistory.put(copy.deepcopy(self.qmaze))
            counts += 1
            self.timer.singleShot(500*counts, self.showCanvas)
            if game_status == 'win':
                self.timer.singleShot(500*(counts+1), self.succeed)
                break
            elif game_status == 'lose':
                self.timer.singleShot(500*(counts+1), self.fail)
                break
    
    def showCanvas(self):
        qmaze = self.mazeHistory.get()
        show(qmaze, ax=self.ax, updateCanvas=self.canvas.draw)

    def showCurCanvas(self):
        self.qmaze.reset(self.qmaze.free_cells[0])
        show(self.qmaze, ax=self.ax, updateCanvas=self.canvas.draw)
                
    def succeed(self):
        QMessageBox.information(self, "Information", "成功解决！")
    
    def fail(self):
        QMessageBox.warning(self, "Warning", "解决失败！")

    def setEpoch(self, str):
        self.label_epoch.setText(str)

    def setLoss(self, str):
        self.label_loss.setText(str)

    def setWinRate(self, str):
        self.label_winRate.setText(str)

    def setTime(self, str):
        self.label_time.setText(str)