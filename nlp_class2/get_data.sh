URL_PREFIX=https://www.clips.uantwerpen.be/conll2000/chunking/
mkdir -p chunking
for suffix in test train
    do
        name="${suffix}.txt.gz" 
        echo "getting ${URL_PREFIX}${name}" 
        wget "${URL_PREFIX}${name}" -O chunking/$name
        gunzip chunking/$name 
    done

