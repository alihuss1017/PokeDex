import os

# ITERATE OVER FILES
for fName in os.listdir('../../cries'):
    # SPLIT NAME AND EXTENSION
    base, ext = os.path.splitext(fName)
    # REMOVE ALL NON 1ST GENERATION SOUND FILES
    if base.isdigit() and int(base) > 151:
        file_path = os.path.join('../../cries', fName)
        os.remove(file_path)

# REMOVE BOTH NIDORAN-F AND NIDORAN-M SOUND FILES
os.remove('../../cries/29.ogg')
os.remove('../../cries/32.ogg')
