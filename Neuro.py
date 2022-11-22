## Reading from a textfile
# readlines()
#
# text_file = open('neurodata.txt')
# # 'r' is facultative
#
# lines = text_file.readlines()
#
# text_file.close()
#
# print (lines)
#
# ## Read
# # the close() function, closes a file descriptor so it no longer refers to anyfile
#
# text_file = open('neurodata.txt')
#
# # 'r' is facultative
#
# print (text_file)
#
# text_file.close()

## Writing a text file, the 'w' is mandatory

# output_file = open('neurodata.txt','w')
# # the 'w' is mandatory
# output_file.write(' number of neuron length: 7/n')
# print(output_file)
# output_file.close()
#
# # Writing and Reading the same file
# file1 = open('neurodata.txt','w')
# file1.write('this is just a dummy test\n')
# file1.close()
#
# file2 = open('neurodata.txt', 'r')
# print (file2)
# file2.close()
#
# file3 = open('neurodata.txt','a')
# file3.write('this is another test\n')
# file3.close()
#
# file4 = open('neurodata.txt', 'r')
# print (file4)
# file4.close()
#
# file5 = open('count.txt', 'w')
# file5.write('this is a final test\n')
# file5.close()
#
# file6 = open('neurodata.txt', 'r')
# print (file6)
# file6.close()
#
#
# file1 = open('neurodata.txt','w')
# file1.write('this is just a dummy test')
# file1.close()
#
# file2 = open('neurodata.txt', 'r')
# variable = file2.read()
#
# print ("Test1: %r" % (variable))
# print ("Test2: %s" % (variable))
# print ("Test3: %30s" % (variable))
# print ("Test4: %-30s" % (variable))
# print ("Test5: %30r" % (variable))
# print ("Test6: %-30r" % (variable))
# print ("Test7: %d, %d, %d" % (1, 2, 3))
# print ("Test8: %2d, %3d, %10d" % (1, 2, 3))
# print ("Test9: %d, %i, %f" % (1, 2, 3))
# print ("Test10: %i, %i, %i" % (1, 2.8, 3.1416))
# print ("Test11: %2i, %5i, %10i" % (1, 2.8, 3.1416))
# print ("Test12: %f, %f, %f" % (1, 2.8, 3.1416))
# print ("Test13: %2f, %2.2f, %10.3f" % (1, 2.8, 3.1416))
# print ("Test14: %2f, %2f, %2f" % (0.11, 10.111, 1000.1111))
# print ("Test15: %2.1f, %2.1f, %2.10f" % (0.11, 10.111, 1000.1111))
#
# file2.close()

#writing a list of number s to a text file2
# data = [16.38, 139.90, 441.46, 29.03, 40.93, 202.07, 142.30, 346.00, 300.00]
#
# out = []
#
# for value in data:
#     out.append(str(value) + '\n')
# open('neurodata.txt', 'w'). writelines(out)
#
# print(data)

# calculating the average from a list of numbers

# calculate average from float numbers
# data = [3.54, 3.48, 3.50, 3.73, 3.40]
# average = sum(data)/ len(data)
# print (average)
#
# # calculating the average from integer numbers
# data = [2,4,5,3,3]
# average = sum(data)/ len(data)
# print(average)

# calculate the median  from a list of numbers
# This one throws me an error:
#LINE(113)TypeError: list indices must be integers or slices, not float
# data = [3.8, 3.40, 4.50, 4.55]
# data.sort()
# mid = len(data)/2
# if len(data) % 2 ==0:
# #    median = [int(data[mid -1] + data[mid] / 2.0)] # ERROR Message list must be integers, no float
# else:
#     median = data[mid]
#
# print(median)

# # Join or concatenate a list
# L = ['1', '2', '3']
# '+'.join(L)
# '1 + 2 + 3'
#
# L = [ 'a', 'b', 'c']
# ''.join(L)
# 'abc'
# L = ['1', '2', '3']
# int('.join(L'))
#123


# Calculate a variance and a standard deviation from a list of numbers

import math

data = [ 3.56, 4.56, 3.57, 6.70, 3.80, 4.60]
average =sum(data)/len(data)
total = 0.0

for value in data:
    total +=(value- average)**2

variance = total / len(data) # population variance
stddev = math.sqrt(variance) # population  stddev

print(variance)
# 1.1807916666666667
print(stddev)
# 1.0866423821417361
