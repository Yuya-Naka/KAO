import os,sys

if __name__ == '__main__':
    outdir = sys.argv[1]

    if not os.path.isdir(outdir):
        sys.exit('%s is not directory' % outdir)

    names = {
        "メッシ":0,
        "ネイマール":1,
        "クリスティアーノロナウド":2,
    }

    exts = ['.jpg','.jpeg']


    for dirpath, dirnames, filename in os.walk(outdir):
        #print(dirnames)
        for dirname in dirnames:
            if dirname in names:
                n = names[dirname]
                member_dir = os.path.join(dirpath,dirname)
                for dirpath2, dirname2, filenames2 in os.walk(member_dir):
                    print(filenames2)
                    if not dirpath2.endswith(dirname):
                        continue
                    #i=0
                    for filename2 in filenames2:
                        #i+=1
                        (fn, ext) = os.path.splitext(filename2)
                        if ext in exts:
                            img_path = os.path.join(dirpath2,filename2)
                            print( '%s %s' % (img_path, n))
                            #print(i)
