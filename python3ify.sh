# replace python 2 syntax to make it python 3 compatible

# change xrange to range
sed 's/xrange/range/' *.py --in-place

# change prints to Python3 version
sed 's/^\(\s*print\)\s\+\(.*\)/\1(\2)/' *.py --in-place