import os

file_list = ['test.txt', 'train.txt', 'valid.txt']
dataset_list = ['icews14', 'icews05-15']

for dataset in dataset_list:
    print(f"Processing {dataset}")
    for filename in file_list:
        with open(os.path.join(dataset, filename), 'r') as f:
            lines = f.read().lower().splitlines()
            split_lines = map(lambda x: x.split("\t"), lines)

            head_list, relation_list, tail_list, time_list = tuple(zip(*split_lines))
            relation_list = ['~'+relation for relation in relation_list]

            result = open(os.path.join(dataset + "_aug", filename[:-4]+".txt"), 'w')
            result.writelines([x+"\n" for x in lines])
            write_lines = []
            for i in range(len(head_list)):
                line = tail_list[i] + "\t" + relation_list[i] + "\t" + head_list[i] + "\t" + time_list[i] + "\n"
                write_lines.append(line)

            result.writelines(write_lines)


# wikidata
print("Processing Wikidata")
for filename in file_list:
    with open(os.path.join("wikidata11k", filename), 'r') as f:
        lines = f.read().lower().splitlines()
        split_lines = map(lambda x: x.split("\t"), lines)

        head_list, relation_list, tail_list, since_list, time_list = tuple(zip(*split_lines))
        inv_relation_list = ['~'+relation for relation in relation_list]

        result = open(os.path.join("wikidata11k_aug", filename[:-4]+".txt"), 'w')

        # result.writelines([x+"\n" for x in lines])
        write_lines = []
        for i in range(len(head_list)):
            line = "\t".join([head_list[i], relation_list[i] + "-" + since_list[i], tail_list[i], time_list[i]]) + "\n"
            write_lines.append(line)
        for i in range(len(head_list)):
            line = "\t".join([tail_list[i], inv_relation_list[i] + "-" + since_list[i], head_list[i], time_list[i]]) + "\n"
            write_lines.append(line)

        result.writelines(write_lines)