import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# x= np.linspace(-1.6, -1.3, 250)

plt.rcParams["figure.figsize"] = [4.50, 3.50]
plt.rcParams["figure.autolayout"] = True

for chain in range(5):

    sam_all = np.load('robertson_dreamzs_5chain_sampled_params_chain_%d_10000.npy' % chain)
    sam_all = sam_all[int(len(sam_all)/2):]
    if chain == 0:
        sam_0 = np.array([x[0] for x in sam_all])
        sam_1 = np.array([x[1] for x in sam_all])
        sam_2 = np.array([x[2] for x in sam_all])
    else:
        sam_0 = np.append(sam_0, np.array([x[0] for x in sam_all]))
        sam_1 = np.append(sam_1, np.array([x[1] for x in sam_all]))
        sam_2 = np.append(sam_2, np.array([x[2] for x in sam_all]))

print(len(sam_0))
print(len(sam_1))
print(len(sam_2))

# scale_factor = 1

plt.figure('sam0')
# n, bins, patches = plt.hist(sam_0, bins= 60, density = True, color='blue', label='histogram_0')
# gaussian_kde_z0 = stats.gaussian_kde(sam_0)
# plt.plot(bins, gaussian_kde_z0(bins) * scale_factor, color= 'springgreen', linewidth= 3, label='kde')
sns.distplot(sam_0, color='springgreen')

plt.figure('sam1')
# n, bins, patches = plt.hist(sam_1, bins= 60, density = True, color='pink', label='histogram_1')
# gaussian_kde_z1 = stats.gaussian_kde(sam_1)
# plt.plot(bins, gaussian_kde_z1(bins) * scale_factor, color= 'yellow', linewidth= 5, label='kde')
sns.distplot(sam_1, color='yellow')


plt.figure('sam2')
# n, bins, patches = plt.hist(sam_2, bins= 60, density = True, color='black', label='histogram_2')
# gaussian_kde_z2 = stats.gaussian_kde(sam_2)
# plt.plot(bins, gaussian_kde_z2(bins) * scale_factor, color= 'magenta', linewidth= 7, label='kde')
sns.distplot(sam_2, color='magenta')


plt.legend(loc=0)
plt.title('Density vs log of Parameter Value')
plt.xlabel('log of parameter value')
plt.ylabel('density')
plt.show()

# sam_0 = np.array([x[0] for x in sam_all if x[0] >= -1.6 and x[0] <= -1.3])
# sam_1 = np.array([x[1] for x in sam_all if x[1] >= 0 and x[1] <= 15])
# sam_2 = np.array([x[2] for x in sam_all if x[2] >= 1 and x[2] <= 7])
#binwidth = (-1.3 - -1.6) / 230
# scale_factor = 1 #len(sam) * binwidth


# R_sqr= -1.6 * np.log(np.array(sam)
# vu = -1.3 * np.pi * np.array(sam)
# s1 = np.sqrt(R_sqr) * np.cos(vu)
# s2 = np.sqrt(R_sqr) * np.sin(vu)
#
# fig = plt.figure(figsize = (12, 4))
# for ind_subplot, zi,  col in zip ((1, 2), (s1, s2), ('crimson', 'dodgerblue')):
#     lalo = fig.add_subplot(1, 2, ind_subplot)
#     plt.hist(zi, bins=40, range=(-1.6, -1.3), color='blue', label='histogram')
#
#     bindwidth =  4 / 250
#     scale_factor = len(zi) * bindwidth
#
#     gaussian_kde_zi = stats.gaussian_kde(s1)
#     plt.plot(x, gaussian_kde_zi(x) * scale_factor, color= 'springgreen', linewidth= 3, label='kde')
#
# plt.show()
    # std_zi = np.std(zi)
    # mean_zi = np.mean(zi))
    # plt.plot(x, stats.norm.pdf((x-mean_zi/std_zi)*scale_factor, color= 'black'))




# Finding the maximun and minimum number of values in the data
# max(sam)
# min(sam)
# print(max(sam))
# print(min(sam))





    # plt.xlim(xmin = -1.6, xmax = -1.3)
    # plt.legend(prop = {'size': 16}, title = 'Map')
    # plt.title('population')
    # plt.set_xlabel('percentage')
    # plt.set_ylabel('number')
    #bindwidth = ((0.8351350471426552-(-3.1179278912226955) / 250))


# print(len(sam))
# print("square root", mt.sqrt)
#print(value_counts(sam))


#logarithm
# logps = np.load('robertson_dreamzs_5chain_logps_chain_0_10000.npy')
# logps = [x[0] for x in logps]
# plt.plot(logps)
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.show()

#logarithm
# logps = np.load('robertson_dreamzs_5chain_logps_chain_1_10000.npy')
# logps = [x[0] for x in logps]
# plt.plot(logps)
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.show()

#sampled_params_chain_0_10000




# density = gaussian_kde(sam)
# plt.hist(sam)
# plt.plot(density(sam), bins=300,)

# plt.style.use('ggplot')


# sam = np.load('robertson_dreamzs_5chain_sampled_params_chain_0_10000.npy')
# sam = [x[0] for x in sam]
# plt.legend(prop={'size': 16}, title = 'Map')
# plt.title('population')
# plt.xlabel('percentage')
# plt.ylabel('number')
# plt.style.use('ggplot')
# gaussian_kde = gaussian_kde(sam)
# plt.hist(sam, bins=210)
# plt.plot(sam, gaussian_kde, bins=300, color='springgreen', linewidth=3, label='kde' )
# plt.show()

#TESTING
# Rsqure = -2 * np.log(sam)
# sam1= np.sqrt(Rsqure)
# bindwidth = 8 / 40
# scale_factor = len(sam1) * bindwidth
#n, x, _ = plt.hist(sam, bins=450, density=True, color='g', edgecolor='k')
#plt.plot(x,gaussian_kde(sam)(x))
# xvalues = np.linspace(-1.6, -1.2,1000)
#plt.xlim(xmin=-1.6, xmax = -1.2)


#sampled_params_chain_0_10000
# sam = np.load('robertson_dreamzs_5chain_sampled_params_chain_1_10000.npy')
# sam = [x[0] for x in sam]
# print(sam)
# plt.xlabel('percentage')
# plt.ylabel('number')
# plt.title('population')
# plt.style.use('ggplot')
# plt.hist(sam, bins=200, color='r', edgecolor='g')
# plt.xlim(xmin=-2, xmax = -1)
# plt.show()













