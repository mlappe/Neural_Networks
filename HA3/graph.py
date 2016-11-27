import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pyplot


x = [1,2,3,4]
y = [3,4,5,6] 

pyplot.plot(x,y)
pyplot.plot(x,[yi +1 for yi in y])

pyplot.savefig('example01.png')
