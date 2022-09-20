import json

# file paths
entity2id_path = "/VL/space/zhan1624/vqar/feature_file/entity2id.json"
entity2id_w = "/VL/space/zhan1624/vqar/feature_file/entity2id_new1.json"
entity2id_w2 = "/VL/space/zhan1624/vqar/feature_file/entity2id_new2.json"
isa_relation_path = "/VL/space/zhan1624/vqar/feature_file/isa_relation.csv"
entity2idrel_w = "/VL/space/zhan1624/vqar/feature_file/entity_rel.json"
with open(entity2id_path) as f_in:
    entity2id_data = json.load(f_in)

def process1():
    """ Process "/" cases in entity2id.

    "/" means space rather than different categories.
    
    Returns:
        write json file (entity2id_new.json)
    """
    def stop_check(input_key):
        """Check ".n" examples and remove them """
        stop_token = [".n.01", ".n.02", ".n.03", ".n.04"]
        if ".n" in input_key:
            for s_t in stop_token:
                if s_t in input_key:
                    return input_key.strip(s_t)
        else:
            return input_key

    new_data = {}
    for key, value in entity2id_data.items():
        if "/" in key:
            key = key.replace("/", "_")
        key = stop_check(key)
        if key not in new_data:
            new_data[key] = value
    with open(entity2id_w, 'w') as f_out:
        json.dump(new_data, f_out, indent=4)

def process2():
    """ entity 'isa' relation
    
    Returns:
        write json file (entity_rel.json and entity2id_new2.json)
        entity2id_new2.json: entity2id file with increased entities
    """
    import csv
    with open(entity2id_w) as f_in:
        entity2id_data = json.load(f_in)
    final_value = entity2id_data[list(entity2id_data.keys())[-1]]
    isa_dict = {}
    with open(isa_relation_path) as csvfile:
        isa_data = csv.reader(csvfile, delimiter="\t")
        for row in isa_data:
            isa_dict[row[1]] = row[2]
    entityrel_dict = {}
    entity2id_data_new = entity2id_data.copy()
    for key, value in entity2id_data.items():
        try:
            entityrel_dict[key] = isa_dict[key]
            if isa_dict[key] not in entity2id_data_new:
                final_value += 1
                entity2id_data_new[isa_dict[key]]=final_value
        except KeyError:
            continue
    
    with open(entity2idrel_w, "w") as f_out1, open(entity2id_w2, 'w') as f_out2:
        json.dump(entityrel_dict, f_out1, indent=4)
        json.dump(entity2id_data_new, f_out2, indent=4)
        

if __name__ == '__main__':
    #process1()
    #process2()
    pass
    
    

