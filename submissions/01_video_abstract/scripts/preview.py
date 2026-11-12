import sys
from PIL import Image
im=Image.open(sys.argv[1]); out=sys.argv[2]; picks=[int(x) for x in sys.argv[3].split(',')]
W=400;sheet=Image.new('RGB',(W*3,W*((len(picks)+2)//3)),'white')
for j,i in enumerate(picks):
    im.seek(i); sheet.paste(im.convert('RGB').resize((W,W)),((j%3)*W,(j//3)*W))
sheet.save(out); print('frames',im.n_frames)
