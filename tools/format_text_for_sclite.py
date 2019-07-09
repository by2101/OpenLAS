import sys

src = sys.argv[1]
dest = sys.argv[2]

fw = open(dest, 'w', encoding="utf8")
with open(src, 'r', encoding='utf8') as f:
  for line in f:
    items = line.strip().split()
    utt = items[0]
    txt = " ".join(items[1:])
    fw.write("{} ({})\n".format(txt, utt))
fw.close()  
  



