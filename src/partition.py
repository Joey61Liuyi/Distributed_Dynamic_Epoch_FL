import numpy as np
from torchvision import datasets, transforms
import random


def randomSplit(M, N, minV, maxV):
    res = []
    while N > 0:
        l = max(minV, M - (N-1)*maxV)
        r = min(maxV, M - (N-1)*minV)
        num = random.randint(l, r)
        N -= 1
        M -= num
        res.append(num)
    print(res)
    return res

def uniform(N, k):
    """Uniform distribution of 'N' items into 'k' groups."""
    dist = []
    avg = N / k
    # Make distribution
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))
    # Return shuffled distribution
    random.shuffle(dist)
    return dist

def normal(N, k):
    """Normal distribution of 'N' items into 'k' groups."""
    dist = []
    # Make distribution
    for i in range(k):
        x = i - (k - 1) / 2
        dist.append(int(N * (np.exp(-x) / (np.exp(-x) + 1)**2)))
    # Add remainders
    remainder = N - sum(dist)
    dist = list(np.add(dist, uniform(remainder, k)))
    # Return non-shuffled distribution
    return dist


def data_organize(idxs_labels, labels):
    data_dict = {}

    labels = np.unique(labels, axis=0)
    for one in labels:
        data_dict[one] = []

    for i in range(len(idxs_labels[1, :])):
        data_dict[idxs_labels[1, i]].append(idxs_labels[0, i])
    return data_dict

def data_partition(training_data, number_of_clients, non_iid_level):

    idxs = np.arange(len(training_data))
    labels = training_data.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    labels = np.unique(labels, axis=0)
    idxs = idxs_labels[0, :]
    data_dict = data_organize(idxs_labels, labels)

    if non_iid_level == 0:
        num_items = int(len(training_data)/number_of_clients)
        data_partition_profile, all_idxs = {}, [i for i in range(len(training_data))]
        for i in range(number_of_clients):
            data_partition_profile[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - data_partition_profile[i])

    else:

        client_dict = {}

        pref_dist = uniform(number_of_clients, len(labels))
        print(pref_dist)
        data_dist = normal(len(training_data), number_of_clients)
        data_dist.sort(reverse=True)
        print(data_dist)

        client_list = list(range(number_of_clients))
        for i in range(len(pref_dist)):
            while pref_dist[i]>0:
                client = np.random.choice(client_list, 1, replace=False)[0]
                client_dict[client] = labels[i]
                pref_dist[i] -= 1
                client_list = list(set(client_list) - set([client]))


        data_partition_profile, all_idxs = {}, [i for i in range(len(training_data))]

        for i in range(number_of_clients):
            pref_number = int(round(data_dist[i] * non_iid_level))

            if pref_number > len(data_dict[client_dict[i]]):
                pref_number = len(data_dict[client_dict[i]])

            data_dist[i] -= pref_number
            data_partition_profile[i] = set(np.random.choice(data_dict[client_dict[i]], pref_number, replace=False))
            all_idxs = list(set(all_idxs) - data_partition_profile[i])
            data_dict[client_dict[i]] = list(set(data_dict[client_dict[i]]) - data_partition_profile[i])

        for i in range(number_of_clients):
            rest_idxs = set(np.random.choice(all_idxs, data_dist[i], replace=False))
            data_partition_profile[i] = set.union(data_partition_profile[i], rest_idxs)
            all_idxs = list(set(all_idxs) - rest_idxs)

    return data_partition_profile

def data_partition_tril(training_data, number_of_clients):

    idxs = np.arange(len(training_data))
    labels = training_data.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    labels = np.unique(labels, axis=0)
    idxs = idxs_labels[0, :]
    data_dict = data_organize(idxs_labels, labels)
    distribution_amount = {}
    for one in data_dict:
        print(len(data_dict[one]), one)
        distribution_amount[one]=uniform(len(data_dict[one]), 4)



    # if non_iid_level == 0:
    #     num_items = int(len(training_data)/number_of_clients)
    #     data_partition_profile, all_idxs = {}, [i for i in range(len(training_data))]
    #     for i in range(number_of_clients):
    #         data_partition_profile[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    #         all_idxs = list(set(all_idxs) - data_partition_profile[i])

    # else:


    assigned_labels = [3,6]
    average_labels = [0,1,2,4,5,7,8,9]


    data_partition_profile, all_idxs = {}, [i for i in range(len(training_data))]

    for i in range(number_of_clients):
        if i == 0:
            data_partition_profile[i] = set(data_dict[3]).union(set(data_dict[6]))
            data_dict[3] = []
            data_dict[6] = []
            pass
        else:
            tep_set = set()
            for one in average_labels:
                pass
                samples = set(np.random.choice(data_dict[one], distribution_amount[one][i-1], replace=False))
                data_dict[one] = list(set(data_dict[one])-samples)
                tep_set = tep_set.union(samples)
                print(distribution_amount[one][i-1])
            print(tep_set)
            data_partition_profile[i] = tep_set
            all_idxs = list(set(all_idxs) - data_partition_profile[i])

    return data_partition_profile
if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 5
    d = data_partition_tril(dataset_train, num)
    print(d)
    for i in d:
        print(len(d[i]))
    # random.seed(0)
    # num_shards = 1200
    # num_users = 50
    # min_shards = round(num_shards/num_users*1/5)
    # max_shards = round(num_shards/num_users*2)
    #
    # a = randomSplit(num_shards,num_users, min_shards,max_shards)
    # print(np.sum(np.array(a)))

