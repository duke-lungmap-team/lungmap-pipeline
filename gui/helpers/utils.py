import ontospy
import pickle

LOCAL_ONTOLOGY_FILE = '../resources/lung_ontology.owl'
PICKLED_ONTOLOGY = '../resources/lung_ontology.pkl'

try:
    # load pickled ontospy object
    f = open(PICKLED_ONTOLOGY, 'rb')
    onto = pickle.load(f)
    f.close()
except FileNotFoundError:
    onto = ontospy.Ontospy(uri_or_path=LOCAL_ONTOLOGY_FILE, rdf_format='xml')

    # pickle the ontology
    f = open(PICKLED_ONTOLOGY, 'wb')
    pickle.dump(onto, f)
    f.close()


def get_onto_protein_uri(ontology, protein_label):
    sparql_proteins_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?p ?p_label WHERE {
    ?p rdfs:subClassOf :Protein .
    ?p :has_synonym ?p_label .
    VALUES ?p_label { "%s" }
}
""" % protein_label

    results = ontology.query(sparql_proteins_query)

    return results


def get_onto_cells_by_protein(ontology, protein_uri):
    sparql_protein_cell_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?c WHERE {
    ?c rdfs:subClassOf* :cell . 
    ?c rdfs:subClassOf ?restriction .
    ?restriction owl:onProperty :has_part ; owl:someValuesFrom ?p .
    VALUES ?p { <%s> }
}
""" % protein_uri

    results = ontology.query(sparql_protein_cell_query)

    return results


def get_onto_tissues_by_cell(ontology, cell_uri):
    sparql_cell_tissue_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?t WHERE {
    ?t rdfs:subClassOf* :tissue .
    ?t rdfs:subClassOf ?restriction .
    ?restriction owl:onProperty :has_part ; owl:someValuesFrom ?c .
    VALUES ?c { <%s> }
}
""" % cell_uri

    results = ontology.query(sparql_cell_tissue_query)

    return results


def get_onto_structures_by_related_uri(ontology, uri):
    sparql_tissue_structure_query1 = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?s ?label ?pred WHERE {
    ?s rdfs:subClassOf* :complex_structure .
    ?s :lungmap_preferred_label ?label . 
    ?s rdfs:subClassOf ?restriction .
    ?restriction owl:onProperty ?pred ; owl:someValuesFrom ?t .
    VALUES ?t { <%s> } .
    VALUES ?pred { :has_part :surrounded_by }
}
""" % uri

    results = ontology.query(sparql_tissue_structure_query1)

    return results


def get_onto_sub_classes(ontology, uri):
    sparql_subclass_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?sub ?label WHERE {
    ?sub rdfs:subClassOf ?uri . 
    ?sub :lungmap_preferred_label ?label . 
    VALUES ?uri { <%s> }
}
""" % uri

    results = ontology.query(sparql_subclass_query)

    return results


def get_probe_structure_map(ontology, probe_labels):
    probe_uri_dict = {}
    probe_structure_dict = {}

    for label in probe_labels:
        label_str = label.replace('Anti-', '')
        uri_result = get_onto_protein_uri(ontology, label_str)
        if len(uri_result) == 0:
            continue
        probe_uri_dict[label] = uri_result[0][0]

    for label, protein_uri in probe_uri_dict.items():
        print(label)
        if label not in probe_structure_dict.keys():
            probe_structure_dict[label] = {'has_part': set(), 'surrounded_by': set()}

        cells = get_onto_cells_by_protein(ontology, protein_uri)

        sub_cells = []

        for cell in cells:
            sub_cells.extend(get_onto_sub_classes(ontology, cell[0]))

        cells.extend(sub_cells)

        for cell in cells:
            print('\t', cell[0].split('#')[1])

            # first check if the cell is directly related to a structure
            structures = get_onto_structures_by_related_uri(ontology, cell[0])

            if len(structures) > 0:
                for structure in structures:
                    print('\t\t\t', structure[0].split('#')[1])
                    rel_type = structure[2].split('#')[1]

                    probe_structure_dict[label][rel_type].add(structure[1].value)

                    sub_structs = get_onto_sub_classes(ontology, structure[0])
                    for ss in sub_structs:
                        print('\t\t\t\tsub_struct: ', ss[0].split('#')[1])
                        probe_structure_dict[label][rel_type].add(ss[1].value)

            tissues = get_onto_tissues_by_cell(ontology, cell[0])

            sub_tissues = []

            for tissue in tissues:
                sub_tissues.extend(get_onto_sub_classes(ontology, tissue[0]))

            tissues.extend(sub_tissues)

            for tissue in tissues:
                print('\t\t', tissue[0].split('#')[1])

                structures = get_onto_structures_by_related_uri(ontology, tissue[0])

                for structure in structures:
                    print('\t\t\t', structure[0].split('#')[1])
                    rel_type = structure[2].split('#')[1]

                    probe_structure_dict[label][rel_type].add(structure[1].value)

                    sub_structs = get_onto_sub_classes(ontology, structure[0])

                    for ss in sub_structs:
                        print('\t\t\t\tsub_struct: ', ss[0].split('#')[1])
                        probe_structure_dict[label][rel_type].add(ss[1].value)

    return probe_structure_dict
