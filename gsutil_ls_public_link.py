import sys

lines = []
for l in sys.stdin:
	lines.append(l)

os = [l.replace("gs://","https://storage.googleapis.com/") for l in lines]
out = "".join(os)
print(out)

