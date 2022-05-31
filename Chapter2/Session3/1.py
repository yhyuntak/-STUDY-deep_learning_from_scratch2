import sys
sys.path.append('../..')

from common.util import preprocess

text = 'YOu say goodbye and I say hello.'
print(preprocess(text))