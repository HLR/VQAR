import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../')

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL

from image_bounding_box_labels import faster_rcnn_bounding_box_labels
from internal_external_rel import internal_external_knowledge_rel

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('vqar_concept') as graph:
    image = Concept(name='image')
    bounding_box = Concept(name='bounding_box')

    ### all bounding box labels as concept name
    ### not very correct here.
    ### now I have all of the concepts from the boundingbox,
    ### however, i also require add all of the entity from the conceptnet into the concept graph.
    for index, data in enumerate(faster_rcnn_bounding_box_labels):
        try:
            locals()[data] = bounding_box(name=data)
            # print(locals()[data], index)
        except:
            pass

    ### image contains bounding_box
    # img_boundingbox_contains = image.contains(bounding_box)
    image.contains(bounding_box)

    ## constraint: is_a relation (concept1, concept2)  for example: is_a (mammal, animal)
    # ifL(pig('x'), mammal('x'))
    # ifL(mammal('x'), animal('x'))
    for key, value in internal_external_knowledge_rel.items():
        locals()[key].is_a(locals()[value])
        ifL(locals()[key]('x'), locals()[value]('x'))

    ## use python graph library to find the following path given a->b. then i will have a->c ### next step
    # ifL(cat('x'), animal('x'))


    # graph.visualize('somewhere.png')
    graph.generate_graphviz_dot('vqar')

    # dot -Tsvg vqar.dot -o output_img.svg ### use command line
    import os
    os.system('dot -Tsvg vqar.dot -o output_img.svg')
    os.system('rm vqar')
    os.system('rm vqar.dot')
    print('Successfully generate the Concept Graph!')


