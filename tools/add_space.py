import sys

src = sys.argv[1]
tgt = sys.argv[2]
fw = open(tgt, 'w', encoding="utf8")
with open(src, 'r', encoding='utf8') as f:
  for line in f:
    items = line.strip().split()
    utt = items[0]
    txt = " ".join(list("".join(items[1:])))
    fw.write(utt + " " + txt + '\n')
fw.close()


