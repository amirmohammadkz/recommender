import pandas as pd
from scipy import spatial
import csv
import time


def compute_cosine(data_frame, item1, item2):
    mini_data_frame = data_frame[(data_frame['item'].isin([item1, item2]))]
    customer_set = list(set(mini_data_frame['customer']))
    item_list1 = []
    item_list2 = []
    print(item1)
    print(item2)
    print(customer_set)
    for customer in customer_set:
        customer_mini_data_frame = data_frame[(data_frame['customer'] == customer)]
        # print(customer_mini_data_frame)
        item1_customer_mini_data_frame = customer_mini_data_frame[(data_frame['item'] == item1)]
        item2_customer_mini_data_frame = customer_mini_data_frame[(data_frame['item'] == item2)]
        item1_customer_rate = list(set(item1_customer_mini_data_frame['rating']))
        item2_customer_rate = list(set(item2_customer_mini_data_frame['rating']))
        if len(item1_customer_rate) == 0:
            item1_customer_rate.append(0.0)
        else:
            item1_customer_rate = [eval(item1_customer_rate[0])]
        if len(item2_customer_rate) == 0:
            item2_customer_rate.append(0.0)
        else:
            item2_customer_rate = [eval(item2_customer_rate[0])]
        # print(item1_customer_rate)
        # print(item2_customer_rate)
        item_list1.append(item1_customer_rate[0])
        item_list2.append(item2_customer_rate[0])
    cosine_res = spatial.distance.cosine(item_list1, item_list2)
    print(cosine_res)
    return cosine_res


def parse(path):
    g = open(path, 'rb')
    reader = csv.reader(g)
    for l in reader:
        # print l
        yield l


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        # print d
        if i == 10000:
            break

    return pd.DataFrame.from_dict(df, orient='index')


times = time.time()
df = getDF('ratings_Books.csv')
df.columns = ['customer', 'item', 'rating', 'timestamp']
print "done"
print(df)
item_list = list(set(df['item']))
customer_list = list(set(df['customer']))
cosine_result = {}
for item in item_list:
    # print(item)
    # print(type(df['item']))
    # print(df['item'])
    this_item_frame = df[(df['item'] == item)]
    # print(this_item_frame)
    item_customer_set = list(set(this_item_frame['customer']))
    # print(item_customer_set)
    item_cosine = {}
    item_item_frame = df[(df['customer'].isin(item_customer_set))]
    # print(item_item_frame)
    item_item_set = list(set(item_item_frame['item']))
    # print(item_item_set)
    item_item_set.remove(item)
    # print("$$$$$$$$$$$$$$$$$$")
    for item_item in item_item_set:
        item_cosine[item_item] = compute_cosine(df, item, item_item)
        print("#############")
    cosine_result[item] = item_cosine
    # print(item_item_frame)
print(cosine_result)

print(item_list.index("0007444117"))
print(len(customer_list))
print time.time() - times
# print(len(set(df[1])))
