import random
from pathlib import Path
import openpyxl as xl


xl.load_workbook

class Dice:

    def roll(self):
        return random.randint(1,6), random.randint(1,6)


dice = Dice()
print(dice.roll())

path = Path()
for file in path.glob('*'):
    print(file)
