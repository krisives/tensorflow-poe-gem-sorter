
import shared, pyautogui, time, tensorflow as tf, numpy as np
from dataclasses import dataclass

shared.fix_crash()
stash_x = 18
stash_y = 128
cell_size = 53
class_names = ['blue', 'empty', 'green', 'red']
model = tf.keras.models.load_model('gem_classifier.hdf5')
screen = pyautogui.screenshot()

@dataclass(repr=True)
class Cell:
    col: int
    row: int
    type: str = 'empty'

class Stash:
    def __init__(self):
        self.cells = []
        for row in range(0, 12):
            for col in range(0, 12):
                self.cells.append(Cell(col, row, 'empty'))

    def get(self, col, row):
        return self.cells[col + row * 12]

    def update(self, cell):
        self.cells[cell.col + cell.row * 12] = cell

    def swap(self, a, b):
        a_col = a.col
        a_row = a.row
        a.col = b.col
        a.row = b.row
        b.col = a_col
        b.row = a_row
        self.update(a)
        self.update(b)

def organize_inventory():
    stash = Stash()

    print("Looking at stash...")
    for col in range(0, 12):
        for row in range(0, 12):
            cell = stash.get(col, row)
            cell.type = predict_cell(col, row)

    filtered = filter(lambda cell: cell.type != 'empty', stash.cells)
    organized = sorted(filtered, key=lambda cell: cell.type)
    print(organized)

    col = 0
    row = 0
    last_type = None
    next = vertical_next
    gap = vertical_gap

    for source in organized:
        if last_type and last_type != source.type:
            col, row = gap(col, row)

        last_type = source.type

        if source.col == col and source.row == row:
            col, row = next(col, row)
            continue

        dest = stash.get(col, row)

        if dest.type == 'empty':
            click_cell(source.col, source.row)
            time.sleep(0.2)
            click_cell(dest.col, dest.row)
            time.sleep(0.2)
        else:
            click_cell(source.col, source.row)
            time.sleep(0.2)
            click_cell(dest.col, dest.row)
            time.sleep(0.2)
            click_cell(source.col, source.row)
            time.sleep(0.2)

        stash.swap(source, dest)
        col, row = next(col, row)

def horizontal_next(col, row):
    col = col + 1

    if col >= 12:
        col = 0
        row = row + 1

    return col, row

def horizontal_gap(col, row):
    return 0, row + 1

def vertical_next(col, row):
    row = row + 1

    if row >= 12:
        row = 0
        col = col + 1

    return col, row

def vertical_gap(col, row):
    return col + 1, 0

def click_cell(col, row):
    x, y = locate_cell(col, row, cell_size // 2)
    pyautogui.moveTo(x, y)
    time.sleep(0.05)
    pyautogui.mouseDown()
    time.sleep(0.05)
    pyautogui.mouseUp()
    time.sleep(0.05)

def locate_cell(col, row, offset=0):
    x = stash_x + col * cell_size + offset
    y = stash_y + row * cell_size + offset
    return x, y

def predict_cell(col, row):
    x, y = locate_cell(col, row)
    cell = screen.crop((x, y, x + cell_size, y + cell_size))
    img_array = tf.keras.preprocessing.image.img_to_array(cell)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    confidence = np.max(score)
    gem = class_names[np.argmax(score)]
    return gem

organize_inventory()
