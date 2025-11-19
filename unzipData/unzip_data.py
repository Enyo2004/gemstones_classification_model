# extract the dataset from the Zip
from zipfile import ZipFile

zip = ZipFile('Gemstones.zip') # provide the path

zip.extractall() # extract the files 

zip.close() # close the zip method



