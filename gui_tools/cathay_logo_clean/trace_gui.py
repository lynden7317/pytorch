import os
import sys
import datetime

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
    import tkFileDialog
else:
    import tkinter as tk
    from tkinter import ttk
    from tkinter.filedialog import askdirectory
    from tkinter.filedialog import asksaveasfilename

import controller


class gui(object):
    def __init__(self):
        super(gui, self).__init__()
        self.root = tk.Tk()
        self.root.title('Trace')
        self.root.minsize(width=700, height=500)  # W=300, H=500
        self.root.maxsize(width=1000, height=800)

        # ==== gui files opts ====
        self.fopts = {}
        self._fopts()
        # =========================

        self.uiCtrl = controller.uiController()

        self.init_ui()

    def init_ui(self):
        self.initGUI()
        self.createNOTE()
        self.createMENU()

        self.createN1Images()

        self.N1FramesPack()

        self.root.mainloop()

    def initGUI(self):
        self.GUI = {}
        self.GUI['MAIN'] = tk.Frame(self.root)
        self.GUI['NB'] = None
        self.GUI['MENU'] = None

        self.GUI['N1'] = None
        self.GUI['N2'] = None
        self.GUI['N3'] = None

        self.GUI['N1_images'] = None
        self.GUI['Log'] = None

        self.GUI['MAIN'].pack(fill=tk.BOTH, expand=True)

    def _fopts(self):
        self.fopts['defaultextension'] = '.txt'
        self.fopts['filetypes'] = [('all files', '.*'), ('text files', '.txt')]
        self.fopts['initialdir'] = 'D:\\'
        self.fopts['initialfile'] = 'myfile.txt'
        self.fopts['parent'] = self.root
        self.fopts['title'] = 'File Operation'

    def createNOTE(self):
        self.GUI['NB'] = ttk.Notebook(self.GUI['MAIN'])

        self.GUI['N1'] = tk.Frame(self.GUI['NB'], width=400, height=100)
        self.GUI['N2'] = tk.Frame(self.GUI['NB'], width=400, height=100)
        self.GUI['N3'] = tk.Frame(self.GUI['NB'], width=400, height=100)

        self.GUI['NB'].add(self.GUI['N1'], text='N1')
        #self.GUI['NB'].add(self.GUI['N2'], text='N2')
        #self.GUI['NB'].add(self.GUI['N3'], text='N3')

        self.GUI['NB'].pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def createMENU(self):
        self.GUI['MENU'] = tk.Menu(self.GUI['MAIN'])
        fileMu = tk.Menu(self.GUI['MENU'], tearoff=0)
        fileMu.add_command(label="開啟目錄", command=self._askdirectory)
        fileMu.add_separator()
        fileMu.add_command(label='儲存檔案', command=self._asksaveasfilename)

        self.GUI['MENU'].add_cascade(label='檔案', menu=fileMu)

        self.root.config(menu=self.GUI['MENU'])

    def createN1Images(self):
        self.GUI['N1_images'] = {'name': "N1_images", 'strVar': tk.StringVar(),
                                 'rad1Var': tk.IntVar(),
                                 'panel': None, 'info': None, 'lab_total': None, 'lab_mark': None}
        
        def rads_default_values():
            self.GUI['N1_images']['rad1Var'].set(0)
        
        self.GUI['N1_images']['setRads2Default'] = rads_default_values
        
        rads_default_values()

        self.GUI['N1_images']['f1'] = tk.LabelFrame(self.GUI['N1'], width=200, height=50,
                                                        pady=5, text='control')
        self.GUI['N1_images']['f2'] = tk.LabelFrame(self.GUI['N1'], width=200, height=50,
                                                        pady=5, text='image/info')
        self.GUI['N1_images']['f3'] = tk.LabelFrame(self.GUI['N1'], width=300, height=500,
                                                    pady=5, text='影像條件')
        uiFrame = self.GUI['N1_images']

        uiFrame['panel'] = tk.Label(uiFrame['f2'], text='Image')
        uiFrame['panel'].grid(row=0, column=0, columnspan=4, padx=1)
        uiFrame['info'] = tk.Label(uiFrame['f2'], text='path', wraplength=150)
        uiFrame['info'].grid(row=0, column=4, columnspan=4, padx=1)

        tk.Label(uiFrame['f1'], text='Total#: ').grid(row=0, column=0)
        uiFrame['lab_total'] = tk.Label(uiFrame['f1'], width=10, bg='white', text='0')
        uiFrame['lab_total'].grid(row=0, column=1)
        tk.Label(uiFrame['f1'], text='Remark#: ').grid(row=0, column=2)
        uiFrame['lab_mark'] = tk.Label(uiFrame['f1'], width=10, bg='white', text='0')
        uiFrame['lab_mark'].grid(row=0, column=3)

        tk.Button(uiFrame['f1'], text='接續處理', width=10, height=1,
                  command=self.uiCtrl.delegator(function='cont', **uiFrame)).grid(row=0, column=4)
        tk.Button(uiFrame['f1'], text='Prev', width=10, height=1,
                  command=self.uiCtrl.delegator(function='prev', **uiFrame)).grid(row=0, column=5)
        tk.Button(uiFrame['f1'], text='Next(Auto. Save)', width=15, height=1, font=('Arial', 12),
                  command=self.uiCtrl.delegator(function='next', **uiFrame)).grid(row=0, column=6)
        #tk.Button(uiFrame['f1'], text='Mark', width=10, height=1,
        #          command=self.uiCtrl.delegator(function='mark', **uiFrame)).grid(row=0, column=7)
        
        
        tk.Label(uiFrame['f3'], text='是否刪除', font=('Arial', 12)).grid(row=0, columnspan=2, sticky=tk.W)
        tk.Radiobutton(uiFrame['f3'], text='否', variable=uiFrame['rad1Var'], value=0, \
        font=('Arial', 12)).grid(row=1, column=0, sticky=tk.W)
        tk.Radiobutton(uiFrame['f3'], text='是', variable=uiFrame['rad1Var'], value=1, \
        font=('Arial', 12)).grid(row=1, column=1, sticky=tk.W)


    def N1FramesPack(self):
        self.GUI['N1_images']['f1'].grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.GUI['N1_images']['f2'].grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.GUI['N1_images']['f3'].grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

    def _askdirectory(self):
        if sys.version_info[0] < 3:
            dirpath = tkFileDialog.askdirectory()
        else:
            dirpath = askdirectory()
        # print 'dir path = ', dirpath
        #print('{} open folder: {}'.format(datetime.now(), dirpath))
        if dirpath not in [None, '']:
            self.uiCtrl.delegator(name='openDir', dirpath=dirpath)()

    def _asksaveasfilename(self):
        try:
            if sys.version_info[0] < 3:
                fpath = tkFileDialog.asksaveasfilename(**self.fopts)
            else:
                fpath = asksaveasfilename(**self.fopts)
            if fpath not in [None, '']:
                self.uiCtrl.delegator(name='fileIO', ftype='save', fpath=fpath)()
        except IOError:
            print('{} Warning: cannot save file {}'.format(datetime.now(), fpath))


def main():
    traceGUI = gui()
    '''
    folderName = 'TraceId/UAT20190927'
    for root, dirs, files in os.walk(folderName):
        #print(root)
        jpgs = []
        for f in files:
            #print('{}/{}'.format(root, f))
    '''

if __name__ == "__main__":
    main()