import sys

if __name__ == '__main__':
	
	if(len(sys.argv)>2):
		path = sys.argv[1]
		out = sys.argv[2]
	else:
		print "Need file to convert and output path"
		exit()


	out = open(out,"w")
	wrote_header = False
	with open(path, "r") as f:
		f.readline()
		while True:
			l = f.readline()
			
			if not l:
				break
			l = l.split()[1:]
			if not wrote_header:
				out.write(",".join(["f_" + str(i) for i in xrange(0, len(l))])+"\n")
				wrote_header = True
			out.write(",".join(l) + "\n")
	out.close()