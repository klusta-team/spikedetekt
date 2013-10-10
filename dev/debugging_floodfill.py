from pylab import *
from spikedetekt.floodfill import connected_components

arr = '''
...xxx.....
...x.x.....
...x..x.x..
...xxx.....
...........
'''

s_back = 1

arr = [line for line in arr.split('\n') if line.strip()]
arr = [[c=='x' for c in line] for line in arr]
arr = array(arr, dtype=int).T

ch_graph = {}
for i in xrange(arr.shape[1]):
    ch_graph[i] = set([j for j in [i-1, i, i+1] if 0<=j<arr.shape[1]])

print 'Original array'
print arr.T
print

components = connected_components(arr, ch_graph, s_back)

colarr = zeros_like(arr)

for c, comp in enumerate(components):
    print 'Component', c
    comparr = zeros_like(arr)
    i, j = zip(*comp)
    comparr[i, j] = 1
    print comparr.T
    print
    colarr[i, j] = c+1
    
subplot(121)
imshow(arr.T, aspect='auto', interpolation='nearest', origin='upper left')
title('mask')
xlabel('Time')
ylabel('Channel')
subplot(122)
imshow(colarr.T, aspect='auto', interpolation='nearest', origin='upper left')
title('components')
xlabel('Time')
ylabel('Channel')
show()
