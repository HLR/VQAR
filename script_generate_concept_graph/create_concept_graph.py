import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../')

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL

import json

### contains all entities, include object class entities and the external source of knowledge graph (conceptnet) entities.
f_entity = open('../feature_file/entity2id_new2.json')
faster_rcnn_bounding_box_labels = json.load(f_entity)

### contains the relations between the object class objects and the conceptnet concepts
f_rel = open('../feature_file/entity_rel.json')
internal_external_knowledge_rel = json.load(f_rel)

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('vqar_concept') as graph:
    image = Concept(name='image')
    bounding_box = Concept(name='bounding_box')

    ### all bounding box labels and the selected external knowledge entities as concept names
    for index, data in enumerate(faster_rcnn_bounding_box_labels):
        try:
            locals()[data] = bounding_box(name=data)
        except:
            pass

    ### image contains bounding_box
    # img_boundingbox_contains = image.contains(bounding_box)
    image.contains(bounding_box)

    ## constraint: is_a relation (concept1, concept2)  for example: is_a (mammal, animal)
    # if logic (is_a constraint) example1: ifL(pig('x'), mammal('x'))
    # if logic (is_a constraint) example2: ifL(mammal('x'), animal('x'))
    for key, value_list in internal_external_knowledge_rel.items(): ### key: string  value: list
        for value in value_list:
            # locals()[key].is_a(locals()[value])
            ifL(locals()[key]('x'), locals()[value]('x'))


    # # graph.visualize('somewhere.png')
    # graph.generate_graphviz_dot('vqar')

    # dot -Tsvg vqar.dot -o output_img.svg ### use command line
    # import os
    # os.system('dot -Tsvg vqar.dot -o output_img.svg')
    # os.system('rm vqar')
    # os.system('rm vqar.dot')
    # print('Successfully generate the Concept Graph!')


