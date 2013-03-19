'''
Run this to generate a Windows installer and Unix .tar.gz distributable file.
The generated files will be put in the dist/ directory.
'''
import os
os.chdir('../.') # work from root
os.system('del MANIFEST')
os.system('setup.py bdist_wininst')
os.system('setup.py sdist --formats=gztar')
