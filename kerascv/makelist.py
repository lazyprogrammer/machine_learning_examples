'''
Use this script to generate a list of all XML files in a folder.
'''

from glob import glob

files = glob('*.xml')
with open('xml_list.txt', 'w') as f:
  for fn in files:
    f.write("%s\n" % fn)